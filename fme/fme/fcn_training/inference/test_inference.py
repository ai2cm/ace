import pathlib
from typing import List, Tuple
import numpy as np
import pytest
import yaml
import dataclasses
from fme.core import metrics
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
    assert len(prediction_ds["timestep"]) == config.n_forward_steps + 1
    for i in range(config.n_forward_steps):
        np.testing.assert_allclose(
            prediction_ds["x"].isel(timestep=i).values + 1,
            prediction_ds["x"].isel(timestep=i + 1).values,
        )
    assert "lat" in prediction_ds.coords
    assert "lon" in prediction_ds.coords


@pytest.mark.parametrize("n_forward_steps,forward_steps_in_memory", [(10, 2), (10, 10)])
def test_inference_writer_boundaries(
    tmp_path: pathlib.Path, n_forward_steps: int, forward_steps_in_memory: int
):
    """Test that data at initial condition boundaires"""
    in_names = ["x"]
    out_names = ["x"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"
    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        n_lat=4,
        n_lon=8,
        nz_interface=4,
    )
    save_plus_one_stepper(
        stepper_path, names=all_names, mean=0.0, std=1.0, data_shape=dim_sizes.shape_2d
    )
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
    )
    config = InferenceConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=n_forward_steps,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=True,
        ),
        validation_data=data.data_loader_params,
        log_video=True,
        save_prediction_files=True,
        forward_steps_in_memory=forward_steps_in_memory,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)
    with mock_wandb() as wandb:
        inference_logs = main(
            yaml_config=str(config_filename),
        )
    # initial condition + n_forward_steps autoregressive steps
    assert len(inference_logs) == config.n_forward_steps + 1
    assert len(wandb.get_logs()) == len(inference_logs)

    prediction_ds = xr.open_dataset(tmp_path / "autoregressive_predictions.nc")
    assert len(prediction_ds["timestep"]) == n_forward_steps + 1
    assert not np.any(np.isnan(prediction_ds["x"].values))

    gen = prediction_ds["x"].sel(source="prediction")
    tar = prediction_ds["x"].sel(source="target")
    gen_time_mean = torch.from_numpy(gen[:, 1:].mean(dim="timestep").values)
    tar_time_mean = torch.from_numpy(tar[:, 1:].mean(dim="timestep").values)
    area_weights = metrics.spherical_area_weights(
        tar["lat"].values, num_lon=len(tar["lon"])
    )
    # check time mean metrics
    assert inference_logs[-1]["inference/mean/forecast_step"] == n_forward_steps
    tol = 1e-4  # relative tolerance
    assert metrics.root_mean_squared_error(
        tar_time_mean, gen_time_mean, area_weights
    ).item() == pytest.approx(inference_logs[-1]["inference/time_mean/rmse/x"], rel=tol)
    assert metrics.weighted_mean_bias(
        tar_time_mean, gen_time_mean, area_weights
    ).item() == pytest.approx(inference_logs[-1]["inference/time_mean/bias/x"], rel=tol)

    prediction_ds = prediction_ds.isel(sample=0)
    ds = xr.open_dataset(data._data_filename)

    # the global initial condition should be identical for prediction and target
    np.testing.assert_allclose(
        prediction_ds["x"].isel(timestep=0).sel(source="prediction").values,
        prediction_ds["x"].isel(timestep=0).sel(source="target").values,
    )
    # the target initial condition should the same as the validation data
    # initial condition
    np.testing.assert_allclose(
        prediction_ds["x"].isel(timestep=0).sel(source="target").values,
        ds["x"].isel(time=0).values,
    )
    for i in range(0, n_forward_steps + 1):
        log = inference_logs[i]
        # metric steps should match timesteps
        assert log["inference/mean/forecast_step"] == i
        gen_i = torch.from_numpy(gen.isel(timestep=i).values)
        tar_i = torch.from_numpy(tar.isel(timestep=i).values)
        # check that manually computed metrics match logged metrics
        assert metrics.root_mean_squared_error(
            tar_i, gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(log["inference/mean/weighted_rmse/x"], rel=tol)
        assert metrics.weighted_mean_bias(
            tar_i, gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(log["inference/mean/weighted_bias/x"], rel=tol)
        assert metrics.gradient_magnitude_percent_diff(
            tar_i, gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(
            log["inference/mean/weighted_grad_mag_percent_diff/x"], rel=tol
        )
        assert metrics.weighted_mean(
            gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(log["inference/mean/weighted_mean_gen/x"], rel=tol)

        # the target obs should be the same as the validation data obs
        np.testing.assert_allclose(
            prediction_ds["x"].isel(timestep=i).sel(source="target").values,
            ds["x"].isel(time=i).values,
        )
        if i > 0:
            timestep_da = prediction_ds["x"].isel(timestep=i)
            # predictions should be previous condition + 1
            np.testing.assert_allclose(
                timestep_da.sel(source="prediction").values,
                prediction_ds["x"].sel(source="prediction").isel(timestep=i - 1).values
                + 1,
            )
            # prediction and target should not have entirely the same values at
            # any timestep > 0
            assert not np.allclose(
                timestep_da.sel(source="prediction").values,
                timestep_da.sel(source="target").values,
            )
