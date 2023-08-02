import pathlib
from typing import List, Tuple
import numpy as np
import pytest
import yaml
import dataclasses
from fme.core.normalizer import FromStateNormalizer
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.testing import FV3GFSData, DimSizes, mock_wandb

from fme.fcn_training.inference.inference import InferenceConfig, main
from fme.fcn_training.registry import ModuleSelector

from fme.fcn_training.train_config import LoggingConfig
import torch
import xarray as xr


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_plus_one_stepper(
    path: pathlib.Path,
    names: List[str],
    mean: float,
    std: float,
    data_shape: Tuple[int, int, int],
):
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": PlusOne()}),
        in_names=["x"],
        out_names=["x"],
        normalization=FromStateNormalizer(
            state={
                "means": {name: mean for name in names},
                "stds": {name: std for name in names},
            }
        ),
        optimization=None,
        prescriber=None,
    )
    stepper = config.get_stepper(
        shapes={
            name: data_shape for name in names
        },  # this data is unused for this test
        max_epochs=0,
    )
    torch.save({"stepper": stepper.get_state()}, path)


@pytest.mark.parametrize("use_prediction_data", [True, False])
def test_inference_plus_one_model(tmp_path: pathlib.Path, use_prediction_data: bool):
    in_names = ["x"]
    out_names = ["x"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"
    dim_sizes = DimSizes(
        n_time=8,
        n_lat=16,
        n_lon=32,
        nz_interface=4,
    )
    if use_prediction_data:
        # use std of 10 so the stepper would have errors at the plus-one problem
        std = 10.0
    else:
        std = 1.0
    save_plus_one_stepper(
        stepper_path, names=all_names, mean=0.0, std=std, data_shape=dim_sizes.shape_2d
    )
    time_varying_values = [float(i) for i in range(dim_sizes.n_time)]
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        time_varying_values=time_varying_values,
    )
    if use_prediction_data:
        prediction_data = data.data_loader_params
    else:
        prediction_data = None
    config = InferenceConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=1,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=True,
        ),
        validation_data=data.data_loader_params,
        prediction_data=prediction_data,
        log_video=True,
        save_prediction_files=True,
        forward_steps_in_memory=1,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        inference_logs = main(
            yaml_config=str(config_filename),
        )
    assert len(inference_logs) == config.n_forward_steps + 1
    assert len(wandb.get_logs()) == len(inference_logs)
    for log in inference_logs:
        # if these are off by something like 90% then probably the stepper
        # is being used instead of the prediction_data
        assert log["inference/mean/weighted_rmse/x"] == 0.0
        assert log["inference/mean/weighted_bias/x"] == 0.0
    prediction_ds = xr.open_dataset(tmp_path / "autoregressive_predictions.nc")
    for i in range(config.n_forward_steps):
        np.testing.assert_allclose(
            prediction_ds["x"].isel(timestep=i).values + 1,
            prediction_ds["x"].isel(timestep=i + 1).values,
        )
