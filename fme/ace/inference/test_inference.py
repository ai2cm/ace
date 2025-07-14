"""Integration test for inference entrypoint."""

import dataclasses
import datetime
import pathlib

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.data_loading.batch_data import PrognosticState
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.inference import (
    InferenceConfig,
    InitialConditionConfig,
    get_initial_condition,
    main,
)
from fme.ace.registry import ModuleSelector
from fme.ace.stepper import SingleModuleStepperConfig
from fme.ace.testing import DimSizes, FV3GFSData
from fme.core.coordinates import (
    DimSize,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.testing import mock_wandb

TIMESTEP = datetime.timedelta(hours=6)


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_stepper(
    path: pathlib.Path,
    in_names: list[str],
    out_names: list[str],
    mean: float,
    std: float,
    horizontal_coords: dict[str, xr.DataArray],
    nz_interface: int,
    timestep: datetime.timedelta = TIMESTEP,
):
    all_names = list(set(in_names).union(out_names))
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": PlusOne()}),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means={name: mean for name in all_names},
            stds={name: std for name in all_names},
        ),
        ocean=OceanConfig(
            surface_temperature_name="sst",
            ocean_fraction_name="ocean_fraction",
        ),
    )
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.tensor(horizontal_coords["lat"].values, dtype=torch.float32),
        lon=torch.tensor(horizontal_coords["lon"].values, dtype=torch.float32),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(nz_interface), bk=torch.arange(nz_interface)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
        variable_metadata={
            "prog": VariableMetadata(
                units="m",
                long_name="a prognostic variable",
            ),
        },
    )
    stepper = config.get_stepper(
        dataset_info=dataset_info,
    )
    torch.save({"stepper": stepper.get_state()}, path)


def test_inference_entrypoint(tmp_path: pathlib.Path):
    forward_steps_in_memory = 2
    # NOTE: number of inputs and outputs has to be the same for the PlusOne
    # stepper module to work properly
    in_names = ["prog", "sst", "forcing_var", "DSWRFtoa"]
    out_names = ["prog", "sst", "ULWRFtoa", "USWRFtoa"]
    stepper_path = tmp_path / "stepper"
    horizontal = [DimSize("lat", 16), DimSize("lon", 32)]
    nz_interface = 4

    dim_sizes = DimSizes(
        n_time=9,
        horizontal=horizontal,
        nz_interface=4,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=["forcing_var", "DSWRFtoa", "sst", "ocean_fraction"],
        dim_sizes=dim_sizes,
        timestep_days=0.25,
        save_vertical_coordinate=False,
    )
    save_stepper(
        stepper_path,
        in_names=in_names,
        out_names=out_names,
        mean=0.0,
        std=1.0,
        horizontal_coords=data.horizontal_coords,
        nz_interface=nz_interface,
    )
    dims = ["sample", "lat", "lon"]
    initial_condition = xr.Dataset(
        {
            "prog": xr.DataArray(
                np.random.rand(2, 16, 32).astype(np.float32), dims=dims
            ),
            "sst": xr.DataArray(
                np.random.rand(2, 16, 32).astype(np.float32), dims=dims
            ),
            "DSWRFtoa": xr.DataArray(
                np.random.rand(2, 16, 32).astype(np.float32), dims=dims
            ),
        }
    )

    initial_condition_path = tmp_path / "init_data" / "ic.nc"
    initial_condition_path.parent.mkdir()
    initial_condition["time"] = xr.DataArray(
        [
            cftime.DatetimeProlepticGregorian(2000, 1, 1, 6),
            cftime.DatetimeProlepticGregorian(2000, 1, 1, 18),
        ],
        dims=["time"],
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
            log_to_wandb=True,
        ),
        initial_condition=InitialConditionConfig(path=str(initial_condition_path)),
        forcing_loader=forcing_loader,
        data_writer=DataWriterConfig(save_prediction_files=True),
        allow_incompatible_dataset=False,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(yaml_config=str(config_filename))
        wandb_logs = wandb.get_logs()

    n_ic_timesteps = 1
    summary_log_step = 1
    assert len(wandb_logs) == n_ic_timesteps + config.n_forward_steps + summary_log_step
    for i, log in enumerate(wandb_logs):
        for metric, val in log.items():
            # check that time series metrics match
            if "inference/mean" in metric:
                if i > 0:
                    assert metric in wandb_logs[i]
                    if np.isnan(val):
                        assert np.isnan(wandb_logs[i][metric])
                    else:
                        assert wandb_logs[i][metric] == val
                elif not np.isnan(val):  # for IC only valid data is reported to wandb
                    assert metric in wandb_logs[i]
                    assert wandb_logs[i][metric] == val

    ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc", decode_timedelta=False
    )
    # prognostic in
    assert "prog" in ds
    assert ds["prog"].attrs == {  # this should come from the stepper metadata
        "units": "m",
        "long_name": "a prognostic variable",
    }
    # diags in
    assert "ULWRFtoa" in ds
    assert ds["ULWRFtoa"].attrs == {  # this should come from default metadata
        "units": "W/m**2",
        "long_name": "Upward LW radiative flux at TOA",
    }
    assert "USWRFtoa" in ds

    # derived in
    assert "net_energy_flux_toa_into_atmosphere" in ds
    assert "units" in ds.net_energy_flux_toa_into_atmosphere.attrs
    assert "long_name" in ds.net_energy_flux_toa_into_atmosphere.attrs
    # forcings not in
    assert "DSWRFtoa" not in ds
    assert "forcing_var" not in ds
    assert ds["prog"].sizes == {"time": 4, "sample": 2, "lat": 16, "lon": 32}
    np.testing.assert_allclose(
        ds["prog"].isel(time=0).values, initial_condition["prog"].values + 1
    )
    np.testing.assert_allclose(
        ds["prog"].isel(time=1).values, ds["prog"].isel(time=0).values + 1, rtol=1e-6
    )
    saved_data = xr.open_dataset(data.data_filename, decode_timedelta=False)
    ops = LatLonCoordinates(
        lat=torch.as_tensor(saved_data["lat"].values.astype(np.float32)),
        lon=torch.as_tensor(saved_data["lon"].values.astype(np.float32)),
    ).get_gridded_operations()
    # check that inference logs match raw output
    for i in range(1, config.n_forward_steps + 1):
        for log_name in wandb_logs[i]:
            if "inference/mean/weighted_mean_gen" in log_name:
                variable_name = log_name.split("/")[-1]
                # note raw output does not include initial condition, hence
                # i-1 below. Code uses area from data, not stepper above.
                raw_variable = ds[variable_name].isel(time=i - 1)
                raw_global_mean = ops.area_weighted_mean(
                    torch.as_tensor(raw_variable.values)
                ).mean()
                np.testing.assert_allclose(
                    raw_global_mean, wandb_logs[i][log_name], rtol=1e-6
                )
    assert "inference/total_steps_per_second" in wandb_logs[-1]


def test_get_initial_condition():
    time_da = xr.DataArray([0, 5], dims=["sample"])
    prognostic_da = xr.DataArray(
        np.random.rand(2, 16, 32), dims=["sample", "lat", "lon"]
    )
    data = xr.Dataset({"prog": prognostic_da, "time": time_da})
    initial_condition = get_initial_condition(data, ["prog"])
    assert isinstance(initial_condition, PrognosticState)
    batch_data = initial_condition.as_batch_data()
    assert batch_data.time.shape == (2, 1)
    initial_times = batch_data.time.isel(time=0)
    assert initial_times.shape == (2,)
    assert initial_times[0] == 0
    assert initial_times[1] == 5
    assert batch_data.data["prog"].shape == (2, 1, 16, 32)
    np.testing.assert_allclose(
        batch_data.data["prog"].squeeze(dim=1).cpu().numpy(), data["prog"].values
    )
    assert batch_data.time.isel(time=0).equals(initial_times)


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


@pytest.mark.parametrize(
    "start_indices, expected_time_values",
    [
        (
            None,
            [
                cftime.datetime(2000, 1, 1, 6),
                cftime.datetime(2000, 1, 1, 12),
                cftime.datetime(2000, 1, 1, 18),
            ],
        ),
        (
            InferenceInitialConditionIndices(
                n_initial_conditions=2, first=0, interval=2
            ),
            [cftime.datetime(2000, 1, 1, 6), cftime.datetime(2000, 1, 1, 18)],
        ),
        (
            ExplicitIndices([0, 2]),
            [cftime.datetime(2000, 1, 1, 6), cftime.datetime(2000, 1, 1, 18)],
        ),
        (
            TimestampList(times=["2000-01-01T06:00:00", "2000-01-01T18:00:00"]),
            [cftime.datetime(2000, 1, 1, 6), cftime.datetime(2000, 1, 1, 18)],
        ),
        (
            ExplicitIndices([1]),
            [
                cftime.datetime(2000, 1, 1, 12),
            ],
        ),
    ],
)
def test__subselect_initial_conditions(tmp_path, start_indices, expected_time_values):
    initial_condition = xr.Dataset(
        {"prog": xr.DataArray(np.random.rand(3, 16, 32), dims=["sample", "lat", "lon"])}
    )
    initial_condition_path = tmp_path / "init_data" / "ic.nc"
    initial_condition_path.parent.mkdir()
    initial_condition["time"] = xr.DataArray(
        [
            cftime.datetime(2000, 1, 1, 6),
            cftime.datetime(2000, 1, 1, 12),
            cftime.datetime(2000, 1, 1, 18),
        ],
        dims=["sample"],
    )
    initial_condition.to_netcdf(initial_condition_path, mode="w")

    ic_config = InitialConditionConfig(
        path=str(initial_condition_path),
        start_indices=start_indices,
    )
    ic_data = ic_config.get_dataset()

    np.testing.assert_array_equal(ic_data.time.values, np.array(expected_time_values))
    assert ic_data["prog"].shape == (len(expected_time_values), 16, 32)
