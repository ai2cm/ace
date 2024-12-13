import dataclasses
import datetime
import os
import pathlib
from typing import List

import dacite
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.aggregator.inference import InferenceEvaluatorAggregatorConfig
from fme.ace.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.ace.inference.evaluator import InferenceEvaluatorConfig, main
from fme.ace.registry import ModuleSelector
from fme.ace.stepper import SingleModuleStepperConfig, TrainOutput
from fme.ace.testing import DimSizes, FV3GFSData, MonthlyReferenceData
from fme.core import metrics
from fme.core.coordinates import DimSize, HybridSigmaPressureCoordinate
from fme.core.dataset.config import XarrayDataConfig
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.ocean import Ocean, OceanConfig
from fme.core.testing import mock_wandb
from fme.core.typing_ import TensorDict, TensorMapping

from .derived_variables import compute_derived_quantities

DIR = pathlib.Path(__file__).parent
TIMESTEP = datetime.timedelta(hours=6)


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_plus_one_stepper(
    path: pathlib.Path,
    in_names: List[str],
    out_names: List[str],
    mean: float,
    std: float,
    data_shape: List[int],
    timestep: datetime.timedelta = TIMESTEP,
    nz_interface: int = 7,
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
    )
    area = torch.ones(data_shape[-2:], device=get_device())
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(nz_interface), bk=torch.arange(nz_interface)
    )
    stepper = config.get_stepper(
        img_shape=(data_shape[-2], data_shape[-1]),
        gridded_operations=LatLonOperations(area),
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
    )
    torch.save({"stepper": stepper.get_state()}, path)


def test_inference_backwards_compatibility(tmp_path: pathlib.Path):
    """
    Inference test using a serialized model from an earlier commit, to ensure
    earlier models can be used with the updated inference code.
    """
    in_names = ["var"]
    out_names = ["var"]
    stepper_path = DIR / "stepper_test_data"

    horizontal = [DimSize("grid_yt", 4), DimSize("grid_xt", 8)]
    dim_sizes = DimSizes(
        n_time=8,
        horizontal=horizontal,
        nz_interface=2,
    )
    std = 1.0
    if not stepper_path.exists():
        # this helps us to re-generate data if the stepper changes
        # to re-generate, just delete the data and run the test (it will fail)
        save_plus_one_stepper(
            stepper_path,
            in_names,
            out_names,
            mean=0.0,
            std=std,
            data_shape=dim_sizes.shape_nd,
        )
        assert False, "stepper_test_data did not exist, it has been created"
    use_prediction_data = False
    n_forward_steps = 2
    inference_helper(
        tmp_path,
        in_names,
        out_names,
        use_prediction_data,
        dim_sizes,
        n_forward_steps,
        stepper_path,
        timestep=TIMESTEP,
    )


@pytest.mark.parametrize(
    "use_prediction_data, n_forward_steps",
    [(True, 2), (False, int(30 / 20 * 36)), (False, 1)],
)
def test_inference_plus_one_model(
    tmp_path: pathlib.Path, use_prediction_data: bool, n_forward_steps: int
):
    in_names = ["var"]
    out_names = ["var"]
    stepper_path = tmp_path / "stepper"

    horizontal = [DimSize("grid_yt", 16), DimSize("grid_xt", 32)]
    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=4,
    )
    if use_prediction_data:
        # use std of 10 so the stepper would have errors at the plus-one problem
        std = 10.0
    else:
        std = 1.0
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=std,
        data_shape=dim_sizes.shape_nd,
        timestep=datetime.timedelta(days=20),
    )
    inference_helper(
        tmp_path,
        in_names,
        out_names,
        use_prediction_data,
        dim_sizes,
        n_forward_steps,
        stepper_path,
        save_monthly_files=False,  # requires timestep == 6h
        timestep=datetime.timedelta(days=20),
    )


def inference_helper(
    tmp_path,
    in_names,
    out_names,
    use_prediction_data,
    dim_sizes: DimSizes,
    n_forward_steps,
    stepper_path,
    timestep: datetime.timedelta,
    save_monthly_files: bool = True,
    derived_names: List[str] = [],
):
    time_varying_values = [float(i) for i in range(dim_sizes.n_time)]
    all_names = list(set(in_names).union(out_names))
    forcing_names = list(set(in_names).difference(out_names))
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        time_varying_values=time_varying_values,
        timestep_days=timestep.total_seconds() / 86400,
        save_vertical_coordinate=False,
    )
    if use_prediction_data:
        prediction_data = data.inference_data_loader_config
    else:
        prediction_data = None

    if dim_sizes.n_time > 365 * 4:
        monthly_reference_filename = str(
            MonthlyReferenceData(
                path=pathlib.Path(tmp_path),
                names=all_names,
                dim_sizes=DimSizes(
                    n_time=48,
                    horizontal=dim_sizes.horizontal,
                    nz_interface=1,
                ),
                n_ensemble=3,
            ).data_filename
        )
    else:
        monthly_reference_filename = None
    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=n_forward_steps,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=True,
        ),
        loader=data.inference_data_loader_config,
        prediction_loader=prediction_data,
        aggregator=InferenceEvaluatorAggregatorConfig(
            monthly_reference_data=monthly_reference_filename, log_video=True
        ),
        data_writer=DataWriterConfig(
            save_prediction_files=True,
            log_extended_video_netcdfs=True,
            save_histogram_files=True,
            save_monthly_files=save_monthly_files,
        ),
        forward_steps_in_memory=1,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(
            yaml_config=str(config_filename),
        )
        wandb_logs = wandb.get_logs()

    all_out_names = out_names + derived_names

    n_ic_timesteps = 1
    summary_log_step = 1
    assert len(wandb_logs) == n_ic_timesteps + config.n_forward_steps + summary_log_step
    for i in range(n_ic_timesteps + config.n_forward_steps):
        log = wandb_logs[i]
        for var in all_out_names:
            if i == 0 and var not in in_names:
                assert f"inference/mean/weighted_rmse/{var}" not in log
            else:
                # if these are off by something like 90% then probably the stepper
                # is being used instead of the prediction_data
                assert log[f"inference/mean/weighted_rmse/{var}"] == 0.0
                assert log[f"inference/mean/weighted_bias/{var}"] == 0.0

    var = list(set(in_names).difference(forcing_names))[0]

    if not use_prediction_data:
        initial_condition_ds = xr.open_dataset(
            tmp_path / "initial_condition.nc", decode_timedelta=False
        )
        for dim_name in ["lat", "lon"]:
            assert dim_name in initial_condition_ds.dims
            assert dim_name in initial_condition_ds.data_vars[var].dims
            assert dim_name in initial_condition_ds.coords

    prediction_ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc",
        decode_timedelta=False,
        decode_times=False,
    )
    assert len(prediction_ds["time"]) == config.n_forward_steps
    for i in range(config.n_forward_steps - 1):
        np.testing.assert_allclose(
            prediction_ds[var].isel(time=i).values + 1,
            prediction_ds[var].isel(time=i + 1).values,
        )
        assert not np.any(np.isnan(prediction_ds[var].isel(time=i + 1).values))

    assert "lat" in prediction_ds.coords
    assert "lon" in prediction_ds.coords

    if use_prediction_data:
        assert not os.path.exists(tmp_path / "restart.nc")
        assert not os.path.exists(tmp_path / "initial_condition.nc")
    else:
        restart_ds = xr.open_dataset(
            tmp_path / "restart.nc", decode_timedelta=False, decode_times=False
        )
        np.testing.assert_allclose(
            prediction_ds[var].isel(time=-1).values,
            restart_ds[var].values,
        )

        ic_ds = xr.open_dataset(
            tmp_path / "initial_condition.nc",
            decode_timedelta=False,
            decode_times=False,
        )
        np.testing.assert_allclose(ic_ds[var].values, 0.0)

    metric_ds = xr.open_dataset(tmp_path / "reduced_autoregressive_predictions.nc")
    assert var in metric_ds.data_vars
    assert metric_ds.data_vars[var].attrs["units"] == "m"
    assert metric_ds.data_vars[var].attrs["long_name"] == f"ensemble mean of {var}"
    assert f"rmse_{var}" in metric_ds.data_vars
    assert metric_ds.data_vars[f"rmse_{var}"].attrs["units"] == "m"
    assert (
        metric_ds.data_vars[f"rmse_{var}"].attrs["long_name"]
        == f"root mean squared error of {var}"
    )
    assert f"bias_{var}" in metric_ds.data_vars
    assert metric_ds.data_vars[f"bias_{var}"].attrs["units"] == "m"
    assert f"min_err_{var}" in metric_ds.data_vars
    assert metric_ds.data_vars[f"min_err_{var}"].attrs["units"] == "m"
    assert f"max_err_{var}" in metric_ds.data_vars
    assert metric_ds.data_vars[f"max_err_{var}"].attrs["units"] == "m"
    assert f"gen_var_{var}" in metric_ds.data_vars
    assert metric_ds.data_vars[f"gen_var_{var}"].attrs["units"] == ""
    assert (
        metric_ds.data_vars[f"gen_var_{var}"].attrs["long_name"]
        == f"prediction variance of {var} as fraction of target variance"
    )
    assert "lat" in metric_ds.coords
    assert "lon" in metric_ds.coords

    time_mean_diagnostics = xr.open_dataset(tmp_path / "time_mean_diagnostics.nc")
    actual_var_names = sorted([str(k) for k in time_mean_diagnostics.keys()])
    assert len(actual_var_names) == 2 * len(all_out_names)
    assert f"bias_map-{var}" in actual_var_names
    assert time_mean_diagnostics.data_vars[f"bias_map-{var}"].attrs["units"] == "m"
    assert f"gen_map-{var}" in actual_var_names
    assert time_mean_diagnostics.data_vars[f"gen_map-{var}"].attrs["units"] == "m"
    assert len(time_mean_diagnostics.coords) == 2
    assert "lat" in time_mean_diagnostics.coords
    assert "lon" in time_mean_diagnostics.coords

    zonal_mean_diagnostics = xr.open_dataset(tmp_path / "zonal_mean_diagnostics.nc")
    actual_var_names = sorted([str(k) for k in zonal_mean_diagnostics.keys()])
    assert len(actual_var_names) == 2 * len(all_out_names)
    assert f"error-{var}" in actual_var_names
    assert zonal_mean_diagnostics.data_vars[f"error-{var}"].attrs["units"] == "m"
    assert f"gen-{var}" in actual_var_names
    assert zonal_mean_diagnostics.data_vars[f"gen-{var}"].attrs["units"] == ""
    assert len(zonal_mean_diagnostics.coords) == 1
    assert "lat" in zonal_mean_diagnostics.coords

    for source in ["target", "prediction"]:
        histograms = xr.open_dataset(tmp_path / f"histograms_{source}.nc")
        actual_var_names = sorted([str(k) for k in histograms.keys()])
        # NOTE: target histograms include forcing variables
        n_vars = (
            len(all_out_names)
            if source == "prediction"
            else len(all_out_names) + len(forcing_names)
        )
        assert len(actual_var_names) == 2 * n_vars
        assert var in actual_var_names
        assert histograms.data_vars[var].attrs["units"] == "count"
        assert f"{var}_bin_edges" in actual_var_names
        assert histograms.data_vars[f"{var}_bin_edges"].attrs["units"] == "m"
        var_counts_per_timestep = histograms[var].sum(dim=["bin"])
        same_count_each_timestep = np.all(
            var_counts_per_timestep.values == var_counts_per_timestep.values[0]
        )
        assert same_count_each_timestep
    if monthly_reference_filename is not None:
        assert f"inference/annual/{var}" in wandb_logs[-1]
        assert f"inference/annual/r2_gen_{var}" in wandb_logs[-1]
        assert f"inference/annual/r2_target_{var}" in wandb_logs[-1]
    assert "inference/total_steps_per_second" in wandb_logs[-1]


@pytest.mark.parametrize(
    "n_forward_steps,forward_steps_in_memory", [(10, 2), (10, 10), (15, 4)]
)
def test_inference_writer_boundaries(
    tmp_path: pathlib.Path, n_forward_steps: int, forward_steps_in_memory: int
):
    """Test that data at initial condition boundaires"""
    in_names = ["var"]
    out_names = ["var"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"

    horizontal = [DimSize("grid_yt", 4), DimSize("grid_xt", 8)]

    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=4,
    )
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        timestep_days=TIMESTEP.total_seconds() / 86400,
    )
    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=n_forward_steps,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(log_to_screen=True, log_to_file=False, log_to_wandb=True),
        loader=data.inference_data_loader_config,
        data_writer=DataWriterConfig(
            save_prediction_files=True,
            time_coarsen=None,
        ),
        forward_steps_in_memory=forward_steps_in_memory,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)
    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(
            yaml_config=str(config_filename),
        )
        inference_logs = wandb.get_logs()
    n_ic_timesteps = 1
    summary_log_step = 1
    assert (
        len(inference_logs)
        == n_ic_timesteps + config.n_forward_steps + summary_log_step
    )

    prediction_ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc", decode_timedelta=False
    )
    target_ds = xr.open_dataset(
        tmp_path / "autoregressive_target.nc", decode_timedelta=False
    )
    # data writers do not include initial condition
    assert len(prediction_ds["time"]) == n_forward_steps
    assert not np.any(np.isnan(prediction_ds["var"].values))

    gen = prediction_ds["var"]
    tar = target_ds["var"]
    gen_time_mean = torch.from_numpy(gen.mean(dim="time").values)
    tar_time_mean = torch.from_numpy(tar.mean(dim="time").values)
    area_weights = metrics.spherical_area_weights(
        tar["lat"].values, num_lon=len(tar["lon"])
    )
    # check time mean metrics
    tol = 1e-4  # relative tolerance
    assert metrics.root_mean_squared_error(
        tar_time_mean, gen_time_mean, area_weights
    ).item() == pytest.approx(
        inference_logs[-1]["inference/time_mean/rmse/var"], rel=tol
    )
    assert metrics.weighted_mean_bias(
        tar_time_mean, gen_time_mean, area_weights
    ).item() == pytest.approx(
        inference_logs[-1]["inference/time_mean/bias/var"], rel=tol
    )

    prediction_ds = prediction_ds.isel(sample=0)
    target_ds = target_ds.isel(sample=0)
    ds = xr.open_dataset(data.data_filename)

    for i in range(0, n_forward_steps):
        # metrics logs includes IC while saved data does not
        log = inference_logs[i + n_ic_timesteps]
        # metric steps should match lead times
        assert log["inference/mean/forecast_step"] == i + n_ic_timesteps
        gen_i = torch.from_numpy(gen.isel(time=i).values)
        tar_i = torch.from_numpy(tar.isel(time=i).values)
        # check that manually computed metrics match logged metrics
        assert metrics.root_mean_squared_error(
            tar_i, gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(log["inference/mean/weighted_rmse/var"], rel=tol)
        assert metrics.weighted_mean_bias(
            tar_i, gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(log["inference/mean/weighted_bias/var"], rel=tol)
        assert metrics.gradient_magnitude_percent_diff(
            tar_i, gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(
            log["inference/mean/weighted_grad_mag_percent_diff/var"], rel=tol
        )
        assert metrics.weighted_mean(
            gen_i, area_weights, dim=(-2, -1)
        ).item() == pytest.approx(log["inference/mean/weighted_mean_gen/var"], rel=tol)

        # the target obs should be the same as the validation data obs
        # ds is original data which includes IC, target_ds does not
        np.testing.assert_allclose(
            target_ds["var"].isel(time=i).values,
            ds["var"].isel(time=i + 1).values,
        )
        if i > 0:
            lead_da = prediction_ds["var"].isel(time=i)
            # predictions should be previous condition + 1
            np.testing.assert_allclose(
                lead_da.values,
                prediction_ds["var"].isel(time=i - 1).values + 1,
            )
            # prediction and target should not have entirely the same values at
            # any lead > 0
            assert not np.allclose(
                lead_da.values,
                target_ds["var"].isel(time=i).values,
            )


def test_inference_data_time_coarsening(tmp_path: pathlib.Path):
    forward_steps_in_memory = 4
    coarsen_factor = 2
    in_names = ["var"]
    out_names = ["var"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"

    horizontal = [DimSize("grid_yt", 16), DimSize("grid_xt", 32)]

    dim_sizes = DimSizes(
        n_time=9,
        horizontal=horizontal,
        nz_interface=4,
    )
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        timestep_days=TIMESTEP.total_seconds() / 86400,
    )
    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=8,
        forward_steps_in_memory=forward_steps_in_memory,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=False,
        ),
        loader=data.inference_data_loader_config,
        data_writer=DataWriterConfig(
            save_prediction_files=True,
            log_extended_video_netcdfs=True,
            time_coarsen=TimeCoarsenConfig(coarsen_factor=coarsen_factor),
            save_histogram_files=True,
        ),
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)
    main(yaml_config=str(config_filename))
    # check that the outputs all have the intended time dimension size
    prediction_ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc", decode_timedelta=False
    )
    n_coarsened_timesteps = config.n_forward_steps // coarsen_factor
    assert (
        len(prediction_ds["time"]) == n_coarsened_timesteps
    ), "raw predictions time dimension size"
    metric_ds = xr.open_dataset(tmp_path / "reduced_autoregressive_predictions.nc")
    assert (
        metric_ds.sizes["timestep"] == n_coarsened_timesteps
    ), "reduced predictions time dimension size"
    for source in ["target", "prediction"]:
        histograms = xr.open_dataset(tmp_path / f"histograms_{source}.nc")
        assert (
            histograms.sizes["time"] == n_coarsened_timesteps
        ), "histograms time dimension size"


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

    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.linspace(0, 1, nz + 1, device=get_device()),
        bk=torch.linspace(0, 1, nz + 1, device=get_device()),
    )

    def derive_func(data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        updated = compute_derived_quantities(
            dict(data),
            vertical_coordinate=vertical_coordinate,
            timestep=TIMESTEP,
            forcing_data=dict(forcing_data),
        )
        return updated

    metrics = {"loss": 42.0}
    fake_data = {k: _make_data() for k in ("gen_data", "target_data")}
    stepped = TrainOutput(
        metrics,
        fake_data["gen_data"],
        fake_data["target_data"],
        time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
        normalize=lambda x: x,
        derive_func=derive_func,
    )

    derived_stepped = stepped.compute_derived_variables()

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

    in_names = ["var", "PRESsfc", "specific_total_water_0", "specific_total_water_1"]
    out_names = ["var", "PRESsfc", "specific_total_water_0", "specific_total_water_1"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"

    horizontal = [DimSize("grid_yt", 16), DimSize("grid_xt", 32)]

    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=3,
    )
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
        nz_interface=dim_sizes.nz_interface,
    )
    time_varying_values = [float(i) for i in range(dim_sizes.n_time)]
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        time_varying_values=time_varying_values,
        timestep_days=TIMESTEP.total_seconds() / 86400,
        num_data_workers=2,
    )
    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=n_forward_steps,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=True,
        ),
        loader=data.inference_data_loader_config,
        prediction_loader=None,
        data_writer=DataWriterConfig(save_prediction_files=True),
        forward_steps_in_memory=1,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(yaml_config=str(config_filename))
        inference_logs = wandb.get_logs()

    # derived variables should not have normalized metrics reported
    assert "inference/mean_norm/weighted_rmse/total_water_path" not in inference_logs[0]
    assert "inference/time_mean_norm/rmse/total_water_path" not in inference_logs[-1]


@pytest.mark.parametrize(
    "time_coarsen,n_forward_steps,forward_steps_in_memory",
    [
        pytest.param(3, 12, 4, id="not_multiple_of_forward_steps_in_memory"),
        pytest.param(-1, 12, 4, id="invalid_time_coarsen"),
        pytest.param(2, 5, 4, id="not_multiple_of_n_forward_steps"),
    ],
)
def test_inference_config_raises_incompatible_timesteps(
    time_coarsen, n_forward_steps, forward_steps_in_memory
):
    base_config_dict = dict(
        experiment_dir="./some_dir",
        n_forward_steps=n_forward_steps,
        checkpoint_path="./some_dir",
        logging=LoggingConfig(),
        loader=InferenceDataLoaderConfig(
            XarrayDataConfig(
                data_path="./some_data",
            ),
            start_indices=InferenceInitialConditionIndices(
                first=0, n_initial_conditions=1, interval=1
            ),
        ),
    )
    base_config_dict["forward_steps_in_memory"] = forward_steps_in_memory
    base_config_dict["data_writer"] = {"time_coarsen": {"coarsen_factor": time_coarsen}}
    with pytest.raises(ValueError):
        dacite.from_dict(
            data_class=InferenceEvaluatorConfig,
            data=base_config_dict,
            config=dacite.Config(strict=True),
        )


def test_inference_ocean_override(tmp_path: pathlib.Path):
    """Test that data at initial condition boundaires"""
    in_names = ["var"]
    out_names = ["var"]
    all_names = list(set(in_names).union(out_names))
    stepper_path = tmp_path / "stepper"
    n_forward_steps = 8

    horizontal = [DimSize("grid_yt", 4), DimSize("grid_xt", 8)]
    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=4,
    )
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=all_names,
        dim_sizes=dim_sizes,
        timestep_days=TIMESTEP.total_seconds() / 86400,
    )
    ocean_override = OceanConfig(
        surface_temperature_name="var",
        ocean_fraction_name="override_ocean_fraction",
    )

    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_forward_steps=n_forward_steps,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(log_to_screen=True, log_to_file=False, log_to_wandb=True),
        loader=data.inference_data_loader_config,
        data_writer=DataWriterConfig(save_prediction_files=True, time_coarsen=None),
        forward_steps_in_memory=4,
        ocean=ocean_override,
    )
    stepper = config.load_stepper()
    assert isinstance(stepper.ocean, Ocean)
    assert (
        stepper.ocean.surface_temperature_name
        == ocean_override.surface_temperature_name
    )
    assert stepper.ocean.ocean_fraction_name == ocean_override.ocean_fraction_name

    stepper_config = config.load_stepper_config()
    assert stepper_config.ocean == ocean_override


def test_inference_timestep_mismatch_error(tmp_path: pathlib.Path):
    """Test that inference with a model trained with a different timestep than
    the forcing data raises an error.
    """
    in_names = ["var"]
    out_names = ["var"]
    stepper_path = tmp_path / "stepper_test_data"

    horizontal = [DimSize("grid_yt", 4), DimSize("grid_xt", 8)]
    dim_sizes = DimSizes(n_time=8, horizontal=horizontal, nz_interface=2)
    std = 1.0
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=std,
        data_shape=dim_sizes.shape_nd,
        timestep=TIMESTEP,
    )
    use_prediction_data = False
    n_forward_steps = 2
    with pytest.raises(ValueError, match="Timestep of the loaded stepper"):
        inference_helper(
            tmp_path,
            in_names,
            out_names,
            use_prediction_data,
            dim_sizes,
            n_forward_steps,
            stepper_path,
            timestep=datetime.timedelta(days=20),
        )


def test_inference_includes_diagnostics(tmp_path: pathlib.Path):
    """Test that diagnostics are included in evaluator metrics and outputs."""
    # NOTE: size of in_names and out_names has to be the same here or the
    # PlusOne outputs won't have the right shape
    in_names = ["prog", "forcing_var", "DSWRFtoa"]
    out_names = ["prog", "ULWRFtoa", "USWRFtoa"]
    stepper_path = tmp_path / "stepper"
    horizontal = [DimSize("grid_yt", 16), DimSize("grid_xt", 32)]
    use_prediction_data = False
    n_forward_steps = 2
    dim_sizes = DimSizes(
        n_time=n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=4,
    )
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
        timestep=datetime.timedelta(days=20),
    )
    inference_helper(
        tmp_path,
        in_names,
        out_names,
        use_prediction_data,
        dim_sizes,
        n_forward_steps,
        stepper_path,
        save_monthly_files=False,  # requires timestep == 6h
        timestep=datetime.timedelta(days=20),
        derived_names=["net_energy_flux_toa_into_atmosphere"],
    )
    ds = xr.open_dataset(
        tmp_path / "autoregressive_predictions.nc",
        decode_timedelta=False,
        decode_times=False,
    )
    # prognostic in
    assert "prog" in ds
    # diags in
    assert "ULWRFtoa" in ds
    assert "USWRFtoa" in ds
    # derived in
    assert "net_energy_flux_toa_into_atmosphere" in ds
    # forcings not in
    assert "DSWRFtoa" not in ds
    assert "forcing_var" not in ds
    # assert only prognostic variables are in initial condition and restart files
    for filename in ["initial_condition.nc", "restart.nc"]:
        ds = xr.open_dataset(tmp_path / filename)
        assert "USWRFtoa" not in ds
        assert "forcing_var" not in ds
        assert "prog" in ds
