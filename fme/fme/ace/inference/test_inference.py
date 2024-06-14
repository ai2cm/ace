"""Integration test for inference entrypoint."""

import dataclasses
import datetime
import pathlib
from typing import List, Tuple

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import fme
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.inference import (
    InferenceConfig,
    InitialConditionConfig,
    get_initial_condition,
    main,
)
from fme.ace.registry import ModuleSelector
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.inference import ForcingDataLoaderConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import FromStateNormalizer
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.testing import DimSizes, FV3GFSData

TIMESTEP = datetime.timedelta(hours=6)


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_stepper(
    path: pathlib.Path,
    in_names: List[str],
    out_names: List[str],
    mean: float,
    std: float,
    data_shape: Tuple[int, int, int],
    timestep: datetime.timedelta = TIMESTEP,
):
    all_names = list(set(in_names).union(out_names))
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": PlusOne()}),
        in_names=in_names,
        out_names=out_names,
        normalization=FromStateNormalizer(
            state={
                "means": {name: mean for name in all_names},
                "stds": {name: std for name in all_names},
            }
        ),
    )
    area = torch.ones(data_shape[-2:], device=fme.get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    stepper = config.get_stepper(
        img_shape=data_shape[-2:],
        area=area,
        sigma_coordinates=sigma_coordinates,
        timestep=timestep,
    )
    torch.save({"stepper": stepper.get_state()}, path)


def test_inference_entrypoint(tmp_path: pathlib.Path):
    forward_steps_in_memory = 2
    in_names = ["prog", "forcing_var"]
    out_names = ["prog"]
    stepper_path = tmp_path / "stepper"
    dim_sizes = DimSizes(
        n_time=9,
        n_lat=16,
        n_lon=32,
        nz_interface=4,
    )
    save_stepper(
        stepper_path,
        in_names=in_names,
        out_names=out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_2d,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=["forcing_var"],
        dim_sizes=dim_sizes,
    )
    initial_condition = xr.Dataset(
        {"prog": xr.DataArray(np.random.rand(2, 16, 32), dims=["sample", "lat", "lon"])}
    )

    initial_condition_path = tmp_path / "init_data" / "ic.nc"
    initial_condition_path.parent.mkdir()
    initial_condition["time"] = xr.DataArray(
        [cftime.datetime(2000, 1, 1, 6), cftime.datetime(2000, 1, 1, 18)],
        dims=["sample"],
    )
    initial_condition.to_netcdf(initial_condition_path, mode="w")
    forcing_loader = ForcingDataLoaderConfig(
        dataset=data.inference_data_loader_config.dataset,
        num_data_workers=data.inference_data_loader_config.num_data_workers,
    )

    config = InferenceConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=4,
        forward_steps_in_memory=forward_steps_in_memory,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=False,
        ),
        initial_condition=InitialConditionConfig(path=str(initial_condition_path)),
        forcing_loader=forcing_loader,
        data_writer=DataWriterConfig(save_prediction_files=True),
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)
    main(yaml_config=str(config_filename))
    ds = xr.open_dataset(tmp_path / "autoregressive_predictions.nc")
    assert "prog" in ds
    assert "forcing_var" not in ds
    assert ds["prog"].sizes == {"time": 4, "sample": 2, "lat": 16, "lon": 32}
    np.testing.assert_allclose(
        ds["prog"].isel(time=0).values, initial_condition["prog"].values + 1
    )
    np.testing.assert_allclose(
        ds["prog"].isel(time=1).values, ds["prog"].isel(time=0).values + 1, rtol=1e-6
    )


def test_get_initial_condition():
    time_da = xr.DataArray([0, 5], dims=["sample"])
    prognostic_da = xr.DataArray(
        np.random.rand(2, 16, 32), dims=["sample", "lat", "lon"]
    )
    data = xr.Dataset({"prog": prognostic_da, "time": time_da})
    initial_condition, initial_times = get_initial_condition(data, ["prog"])
    assert initial_times.shape == (2,)
    assert initial_times[0] == 0
    assert initial_times[1] == 5
    assert initial_condition["prog"].shape == (2, 16, 32)
    np.testing.assert_allclose(initial_condition["prog"].numpy(), data["prog"].values)
    assert initial_condition["prog"].device == fme.get_device()


def test_get_initial_condition_raises_bad_variable_shape():
    time_da = xr.DataArray([0, 5], dims=["sample"])
    prognostic_da = xr.DataArray(
        np.random.rand(2, 8, 16, 32), dims=["sample", "layer", "lat", "lon"]
    )
    data = xr.Dataset({"prog": prognostic_da, "time": time_da})
    with pytest.raises(ValueError):
        get_initial_condition(data, ["prog"])


def test_get_initial_condition_raises_missing_time():
    prognostic_da = xr.DataArray(
        np.random.rand(2, 16, 32), dims=["sample", "lat", "lon"]
    )
    data = xr.Dataset({"prog": prognostic_da})
    with pytest.raises(ValueError):
        get_initial_condition(data, ["prog"])


def test_get_initial_condition_raises_mismatched_time_length():
    time_da = xr.DataArray([0, 5, 10], dims=["time"])
    prognostic_da = xr.DataArray(
        np.random.rand(2, 16, 32), dims=["sample", "lat", "lon"]
    )
    data = xr.Dataset({"prog": prognostic_da, "time": time_da})
    with pytest.raises(ValueError):
        get_initial_condition(data, ["prog"])
