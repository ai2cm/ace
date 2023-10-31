import dataclasses
import pathlib
from typing import List, Tuple

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.core import metrics
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.device import get_device
from fme.core.normalizer import FromStateNormalizer
from fme.core.stepper import SingleModuleStepperConfig, SteppedData
from fme.core.testing import DimSizes, FV3GFSData, mock_wandb
from fme.fcn_training.inference.derived_variables import compute_derived_quantities
from fme.fcn_training.inference.inference import InferenceConfig, main
from fme.fcn_training.registry import ModuleSelector
from fme.fcn_training.train_config import LoggingConfig

DIR = pathlib.Path(__file__).parent


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
        prescriber=None,
    )
    area = torch.ones(data_shape[-2:], device=get_device())
    sigma_coordinates = SigmaCoordinates(ak=torch.arange(7), bk=torch.arange(7))
    stepper = config.get_stepper(
        shapes={name: data_shape for name in names},
        area=area,
        sigma_coordinates=sigma_coordinates,
    )
    torch.save({"stepper": stepper.get_state()}, path)


def test_inference_backwards_compatibility(tmp_path: pathlib.Path):
    """
    Inference test using a serialized model from an earlier commit, to ensure
    earlier models can be used with the updated inference code.
    """
    in_names = ["x"]
    out_names = ["x"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = DIR / "stepper_test_data"
    dim_sizes = DimSizes(
        n_time=8,
        n_lat=4,
        n_lon=8,
        nz_interface=2,
    )
    std = 1.0
    if not stepper_path.exists():
        # this helps us to re-generate data if the stepper changes
        # to re-generate, just delete the data and run the test (it will fail)
        save_plus_one_stepper(
            stepper_path,
            names=all_names,
            mean=0.0,
            std=std,
            data_shape=dim_sizes.shape_2d,
        )
        assert False, "stepper_test_data did not exist, it has been created"
    use_prediction_data = False
    n_forward_steps = 2
    inference_helper(
        tmp_path,
        all_names,
        use_prediction_data,
        dim_sizes,
        n_forward_steps,
        stepper_path,
    )


@pytest.mark.parametrize(
    "use_prediction_data, n_forward_steps",
    [(True, 2), (True, 1), (False, 2), (False, 1)],
)
def test_inference_plus_one_model(
    tmp_path: pathlib.Path, use_prediction_data: bool, n_forward_steps: int
):
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
    inference_helper(
        tmp_path,
        all_names,
        use_prediction_data,
        dim_sizes,
        n_forward_steps,
        stepper_path,
    )


def inference_helper(
    tmp_path, all_names, use_prediction_data, dim_sizes, n_forward_steps, stepper_path
):
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
        n_forward_steps=n_forward_steps,
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
        log_extended_video_netcdfs=True,
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
    prediction_ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc", decode_timedelta=False
    )
    assert len(prediction_ds["lead"]) == config.n_forward_steps + 1
    for i in range(config.n_forward_steps):
        np.testing.assert_allclose(
            prediction_ds["x"].isel(lead=i).values + 1,
            prediction_ds["x"].isel(lead=i + 1).values,
        )
    assert "lat" in prediction_ds.coords
    assert "lon" in prediction_ds.coords
    metric_ds = xr.open_dataset(tmp_path / "reduced_autoregressive_predictions.nc")
    assert "x" in metric_ds.data_vars
    assert metric_ds.data_vars["x"].attrs["units"] == "m"
    assert metric_ds.data_vars["x"].attrs["long_name"] == "ensemble mean of x"
    assert "rmse_x" in metric_ds.data_vars
    assert metric_ds.data_vars["rmse_x"].attrs["units"] == "m"
    assert (
        metric_ds.data_vars["rmse_x"].attrs["long_name"]
        == "root mean squared error of x"
    )
    assert "bias_x" in metric_ds.data_vars
    assert metric_ds.data_vars["bias_x"].attrs["units"] == "m"
    assert "min_err_x" in metric_ds.data_vars
    assert metric_ds.data_vars["min_err_x"].attrs["units"] == "m"
    assert "max_err_x" in metric_ds.data_vars
    assert metric_ds.data_vars["max_err_x"].attrs["units"] == "m"
    assert "gen_var_x" in metric_ds.data_vars
    assert metric_ds.data_vars["gen_var_x"].attrs["units"] == ""
    assert (
        metric_ds.data_vars["gen_var_x"].attrs["long_name"]
        == "prediction variance of x as fraction of target variance"
    )
    assert "lat" in metric_ds.coords
    assert "lon" in metric_ds.coords

    time_mean_diagnostics = xr.open_dataset(tmp_path / "time_mean_diagnostics.nc")
    actual_var_names = sorted([str(k) for k in time_mean_diagnostics.keys()])
    assert len(actual_var_names) == 2
    assert "bias_map-x" in actual_var_names
    assert time_mean_diagnostics.data_vars["bias_map-x"].attrs["units"] == "m"
    assert "gen_map-x" in actual_var_names
    assert time_mean_diagnostics.data_vars["gen_map-x"].attrs["units"] == ""
    assert len(time_mean_diagnostics.coords) == 2
    assert "lat" in time_mean_diagnostics.coords
    assert "lon" in time_mean_diagnostics.coords

    zonal_mean_diagnostics = xr.open_dataset(tmp_path / "zonal_mean_diagnostics.nc")
    actual_var_names = sorted([str(k) for k in zonal_mean_diagnostics.keys()])
    assert len(actual_var_names) == 2
    assert "error-x" in actual_var_names
    assert zonal_mean_diagnostics.data_vars["error-x"].attrs["units"] == "m"
    assert "gen-x" in actual_var_names
    assert zonal_mean_diagnostics.data_vars["gen-x"].attrs["units"] == ""
    assert len(zonal_mean_diagnostics.coords) == 1
    assert "lat" in zonal_mean_diagnostics.coords

    histograms = xr.open_dataset(tmp_path / "histograms.nc")
    actual_var_names = sorted([str(k) for k in histograms.keys()])
    assert len(actual_var_names) == 2
    assert "x" in actual_var_names
    assert histograms.data_vars["x"].attrs["units"] == "count"
    assert "x_bin_edges" in actual_var_names
    assert histograms.data_vars["x_bin_edges"].attrs["units"] == "m"
    x_counts_per_timestep = histograms["x"].sum(dim=["bin", "source"])
    same_count_each_timestep = np.all(
        x_counts_per_timestep.values == x_counts_per_timestep.values[0]
    )
    assert same_count_each_timestep


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

    prediction_ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc", decode_timedelta=False
    )
    assert len(prediction_ds["lead"]) == n_forward_steps + 1
    assert not np.any(np.isnan(prediction_ds["x"].values))

    gen = prediction_ds["x"].sel(source="prediction")
    tar = prediction_ds["x"].sel(source="target")
    gen_time_mean = torch.from_numpy(gen[:, 1:].mean(dim="lead").values)
    tar_time_mean = torch.from_numpy(tar[:, 1:].mean(dim="lead").values)
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
        prediction_ds["x"].isel(lead=0).sel(source="prediction").values,
        prediction_ds["x"].isel(lead=0).sel(source="target").values,
    )
    # the target initial condition should the same as the validation data
    # initial condition
    np.testing.assert_allclose(
        prediction_ds["x"].isel(lead=0).sel(source="target").values,
        ds["x"].isel(time=0).values,
    )
    for i in range(0, n_forward_steps + 1):
        log = inference_logs[i]
        # metric steps should match lead times
        assert log["inference/mean/forecast_step"] == i
        gen_i = torch.from_numpy(gen.isel(lead=i).values)
        tar_i = torch.from_numpy(tar.isel(lead=i).values)
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
            prediction_ds["x"].isel(lead=i).sel(source="target").values,
            ds["x"].isel(time=i).values,
        )
        if i > 0:
            lead_da = prediction_ds["x"].isel(lead=i)
            # predictions should be previous condition + 1
            np.testing.assert_allclose(
                lead_da.sel(source="prediction").values,
                prediction_ds["x"].sel(source="prediction").isel(lead=i - 1).values + 1,
            )
            # prediction and target should not have entirely the same values at
            # any lead > 0
            assert not np.allclose(
                lead_da.sel(source="prediction").values,
                lead_da.sel(source="target").values,
            )


@pytest.mark.parametrize("has_required_fields", [True, False])
def test_compute_derived_quantities(has_required_fields):
    """Checks that tensors are added to the data dictionary appropriately."""
    n_sample, n_time, nx, ny, nz = 2, 3, 4, 5, 6

    def _make_data():
        vars = ["a"]
        if has_required_fields:
            additional_fields = [
                "specific_total_water_{}".format(i) for i in range(nz)
            ] + ["PRESsfc"]
            vars += additional_fields
        return {
            var: torch.randn(n_sample, n_time, nx, ny, device=get_device())
            for var in vars
        }

    loss = 42.0
    fake_data = {
        k: _make_data()
        for k in ("gen_data", "target_data", "gen_data_norm", "target_data_norm")
    }
    stepped = SteppedData(
        loss,
        fake_data["gen_data"],
        fake_data["target_data"],
        fake_data["gen_data_norm"],
        fake_data["target_data_norm"],
    )

    sigma_coords = SigmaCoordinates(
        ak=torch.linspace(0, 1, nz + 1, device=get_device()),
        bk=torch.linspace(0, 1, nz + 1, device=get_device()),
    )
    derived_stepped = compute_derived_quantities(stepped, sigma_coords)

    dry_air_name = "surface_pressure_due_to_dry_air"
    water_path_name = "total_water_path"
    existence_check = (
        dry_air_name in derived_stepped.gen_data
        and dry_air_name in derived_stepped.target_data
        and water_path_name in derived_stepped.gen_data
        and water_path_name in derived_stepped.target_data
    )

    if has_required_fields:
        assert existence_check
        fields = (
            derived_stepped.gen_data[dry_air_name],
            derived_stepped.target_data[dry_air_name],
            derived_stepped.gen_data["a"],
            derived_stepped.target_data["a"],
        )
        for f in fields:
            assert f.shape == (n_sample, n_time, nx, ny)
    else:
        assert not existence_check


def test_derived_metrics_run_without_errors(tmp_path: pathlib.Path):
    """Checks that derived metrics are computed during inferece without errors."""

    n_forward_steps = 2

    in_names = ["x", "PRESsfc", "specific_total_water_0", "specific_total_water_1"]
    out_names = ["x", "PRESsfc", "specific_total_water_0", "specific_total_water_1"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"
    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        n_lat=16,
        n_lon=32,
        nz_interface=4,
    )
    save_plus_one_stepper(
        stepper_path, names=all_names, mean=0.0, std=1.0, data_shape=dim_sizes.shape_2d
    )
    time_varying_values = [float(i) for i in range(dim_sizes.n_time)]
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        time_varying_values=time_varying_values,
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
        prediction_data=None,
        log_video=True,
        save_prediction_files=True,
        forward_steps_in_memory=1,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as _:
        _ = main(
            yaml_config=str(config_filename),
        )
