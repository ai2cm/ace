import datetime
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference import (
    AnnualMetricConfig,
    EnsoCoefficientMetricConfig,
    EnsoIndexMetricConfig,
    HierarchicalInferenceEvaluatorAggregatorConfig,
    HistogramMetricConfig,
    InferenceEvaluatorAggregatorConfig,
    LegacyFlagInferenceEvaluatorAggregatorConfig,
    MeanMetricConfig,
    PowerSpectrumMetricConfig,
    SeasonalMetricConfig,
    StepMeanEntry,
    StepMeanMetricConfig,
    TimeMeanMetricConfig,
    VideoMetricConfig,
    ZonalMeanMetricConfig,
)
from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.wandb import Image

TIMESTEP = datetime.timedelta(hours=6)
LOG_ZONAL_MEAN_IMAGES = 100
DATA_DIR = pathlib.Path(__file__).parent / "testdata"


def get_ds_info(nx: int, ny: int) -> DatasetInfo:
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lon=torch.arange(nx),
            lat=torch.arange(ny),
        ),
        timestep=TIMESTEP,
    )


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ns]"), dims=dims)


def logs_to_raw(
    logs: dict[str, float | Image], max_size: int
) -> dict[str, float | torch.Tensor]:
    raw_logs = {}
    for key, value in logs.items():
        if isinstance(value, Image):
            assert value.image is not None
            data = torch.as_tensor(np.array(value.image))
            if data.shape[0] * data.shape[1] < max_size:
                raw_logs[key] = torch.as_tensor(np.array(value.image))
        elif isinstance(value, plt.Figure):
            pass  # not comparable
        else:
            raw_logs[key] = value
    return raw_logs


def regress_logs(
    logs: dict[str, float | Image | plt.Figure],
    label: str,
    max_size: int,
    path: pathlib.Path = DATA_DIR,
):
    label = label.replace("/", "-")
    raw_logs = logs_to_raw(logs, max_size=max_size)
    regression_file = path / f"{label}-regression.pt"
    if not regression_file.exists():
        for key, value in logs.items():
            if isinstance(value, Image):
                assert value.image is not None
                shape = np.array(value.image).shape
                if shape[0] * shape[1] < max_size:
                    value.image.save(path / f"{label}-{key.replace('/', '-')}.png")
            else:
                pass
        torch.save(raw_logs, regression_file)
        pytest.fail(
            f"Regression file {regression_file} did not exist, so it was created"
        )
    else:
        raw_logs_loaded = torch.load(regression_file, map_location="cpu")
        assert set(raw_logs.keys()) == set(raw_logs_loaded.keys())
        for key, value in raw_logs.items():
            value = value.cpu() if isinstance(value, torch.Tensor) else value
            torch.testing.assert_close(
                value, raw_logs_loaded[key], rtol=1e-3, atol=1e-3
            )


def test_logs_regression():
    torch.manual_seed(0)
    n_sample = 10
    n_time = 24
    nx = 90
    ny = 45
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    agg = InferenceEvaluatorAggregatorConfig(
        metrics=[
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, target="norm"),
            StepMeanMetricConfig(step=4, name="one_day_mean", target="denorm"),
            StepMeanMetricConfig(step=4, name="one_day_mean_norm", target="norm"),
            PowerSpectrumMetricConfig(),
            ZonalMeanMetricConfig(zonal_mean_max_size=LOG_ZONAL_MEAN_IMAGES),
            TimeMeanMetricConfig(target="denorm"),
            TimeMeanMetricConfig(target="norm"),
        ],
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    time = xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"])

    logs = agg.record_batch(
        data=PairedData.new_on_device(
            prediction={"a": torch.randn(n_sample, n_time, ny, nx).to(get_device())},
            reference={"a": torch.randn(n_sample, n_time, ny, nx).to(get_device())},
            time=time,
            labels=None,
        ),
    )
    assert len(logs) == n_time
    expected_step_keys = [
        "mean/forecast_step",
        "mean/weighted_mean_gen/a",
        "mean/weighted_mean_target/a",
        "mean/weighted_rmse/a",
        "mean/weighted_std_gen/a",
        "mean/weighted_bias/a",
        "mean/weighted_grad_mag_percent_diff/a",
        "mean_norm/forecast_step",
        "mean_norm/weighted_mean_gen/a",
        "mean_norm/weighted_mean_target/a",
        "mean_norm/weighted_rmse/a",
        "mean_norm/weighted_std_gen/a",
        "mean_norm/weighted_bias/a",
    ]
    for log in logs:
        for key in expected_step_keys:
            assert key in log, key
        assert len(log) == len(expected_step_keys), set(log).difference(
            expected_step_keys
        )

    summary_logs = agg.get_summary_logs()
    for key, value in summary_logs.items():
        if not isinstance(value, float | Image | plt.Figure):
            pytest.fail(
                f"Summary log {key} is of type {type(value)}, "
                "not a float or Image or plt.Figure"
            )
    if get_device() == torch.device("cpu"):
        regress_logs(
            summary_logs,
            label="test_evaluator-test_logs_labels_exist",
            max_size=6 * ny * nx,
        )
    expected_keys = [
        "mean_step_20/weighted_rmse/a",
        "mean_step_20/weighted_bias/a",
        "mean_step_20/weighted_grad_mag_percent_diff/a",
        "mean_step_20_norm/weighted_rmse/a",
        "mean_step_20_norm/weighted_rmse/channel_mean",
        "one_day_mean/weighted_rmse/a",
        "one_day_mean/weighted_bias/a",
        "one_day_mean/weighted_grad_mag_percent_diff/a",
        "one_day_mean_norm/weighted_rmse/a",
        "one_day_mean_norm/weighted_rmse/channel_mean",
        "power_spectrum/a",
        "power_spectrum/negative_norm_bias/a",
        "power_spectrum/positive_norm_bias/a",
        "power_spectrum/mean_abs_norm_bias/a",
        "power_spectrum/smallest_scale_norm_bias/a",
        "time_mean/rmse/a",
        "time_mean/bias/a",
        "time_mean/bias_map/a",
        "time_mean/gen_map/a",
        "time_mean_norm/rmse/a",
        "time_mean_norm/gen_map/a",
        "time_mean_norm/rmse/channel_mean",
        "zonal_mean/error/a",
        "zonal_mean/gen/a",
    ]
    assert set(summary_logs.keys()) == set(expected_keys)


def test_inference_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 90
    ny = 45
    ds_info = get_ds_info(nx, ny)
    initial_time = (get_zero_time(shape=[n_sample, 0], dims=["sample", "time"]),)
    agg = InferenceEvaluatorAggregatorConfig(
        metrics=[
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, target="norm"),
            StepMeanMetricConfig(step=4, name="one_day_mean", target="denorm"),
            StepMeanMetricConfig(step=4, name="one_day_mean_norm", target="norm"),
            PowerSpectrumMetricConfig(),
            ZonalMeanMetricConfig(zonal_mean_max_size=LOG_ZONAL_MEAN_IMAGES),
            VideoMetricConfig(),
            TimeMeanMetricConfig(target="denorm"),
            TimeMeanMetricConfig(target="norm"),
        ],
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    logs = agg.record_batch(
        data=PairedData.new_on_device(
            prediction={
                "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
            },
            reference={"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())},
            time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
            labels=None,
        ),
    )
    assert isinstance(logs, list)
    assert len(logs) == n_time
    assert "mean/weighted_bias/a" in logs[0]
    assert "mean/weighted_mean_gen/a" in logs[0]
    assert "mean/weighted_mean_target/a" in logs[0]
    assert "mean/weighted_grad_mag_percent_diff/a" in logs[0]
    assert "mean/weighted_rmse/a" in logs[0]
    assert "mean_norm/weighted_bias/a" in logs[0]
    assert "mean_norm/weighted_mean_gen/a" in logs[0]
    assert "mean_norm/weighted_mean_target/a" in logs[0]
    assert "mean_norm/weighted_rmse/a" in logs[0]
    # series/table data should be rolled out, not included as a table
    assert "mean/series" not in logs[0]
    assert "mean_norm/series" not in logs[0]
    assert "reduced/series" not in logs[0]
    assert "reduced_norm/series" not in logs[0]


@pytest.mark.parametrize(
    "window_len, n_windows",
    [
        pytest.param(3, 1, id="single_window"),
        pytest.param(3, 2, id="two_windows"),
    ],
)
def test_inference_logs_length(window_len: int, n_windows: int):
    """
    Test that the inference logs are the correct length when using one or more
    windows.
    """
    nx, ny = 4, 4
    ds_info = get_ds_info(nx, ny)
    initial_time = (get_zero_time(shape=[2, 0], dims=["sample", "time"]),)
    n_forward_steps = window_len * n_windows - 1
    step_mean_metrics: list = (
        []
        if n_forward_steps < 20
        else [
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, target="norm"),
        ]
    )
    agg = InferenceEvaluatorAggregatorConfig(
        metrics=[
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
            *step_mean_metrics,
            PowerSpectrumMetricConfig(),
            ZonalMeanMetricConfig(zonal_mean_max_size=LOG_ZONAL_MEAN_IMAGES),
            TimeMeanMetricConfig(target="denorm"),
            TimeMeanMetricConfig(target="norm"),
        ],
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_forward_steps,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    target_data = BatchData.new_on_device(
        data={"a": torch.zeros([2, window_len, ny, nx], device=get_device())},
        time=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
        labels=None,
        horizontal_dims=["lat", "lon"],
    )
    i_start = 0
    for i in range(n_windows):
        sample_data = {"a": torch.zeros([2, window_len, ny, nx], device=get_device())}
        for i in range(window_len):
            sample_data["a"][..., i, :, :] = float(i_start + i)
        paired_data = PairedData.new_on_device(
            prediction=sample_data,
            reference=target_data.data,
            time=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
            labels=None,
        )
        logs = agg.record_batch(
            data=paired_data,
        )
        assert len(logs) == window_len
        i_start += window_len


def test_flush_diagnostics(tmpdir):
    nx, ny, n_sample, n_time = 2, 2, 10, 21
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])
    agg = InferenceEvaluatorAggregatorConfig(
        metrics=[
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, name="mean_step_20_norm", target="norm"),
            PowerSpectrumMetricConfig(),
            ZonalMeanMetricConfig(zonal_mean_max_size=LOG_ZONAL_MEAN_IMAGES),
            TimeMeanMetricConfig(target="denorm"),
            TimeMeanMetricConfig(target="norm"),
            VideoMetricConfig(),
            HistogramMetricConfig(),
        ],
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        output_dir=tmpdir,
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    agg.record_batch(
        data=PairedData(
            prediction=gen_data,
            reference=target_data,
            time=time,
            labels=None,
        ),
    )
    agg.flush_diagnostics()
    expected_files = [  # note: time-dependent aggregators not tested here
        "mean",
        "mean_norm",
        "mean_step_20",
        "mean_step_20_norm",
        "power_spectrum",
        "zonal_mean",
        "time_mean",
        "time_mean_norm",
        "histogram",
        "video",
    ]
    for file in expected_files:
        assert (tmpdir / f"{file}_diagnostics.nc").exists()


def test_agg_raises_without_output_dir():
    ds_info = get_ds_info(nx=2, ny=2)
    with pytest.raises(
        ValueError, match="Output directory must be set to save diagnostics"
    ):
        InferenceEvaluatorAggregatorConfig(
            metrics=[
                MeanMetricConfig(target="denorm"),
                MeanMetricConfig(target="norm"),
                PowerSpectrumMetricConfig(),
                ZonalMeanMetricConfig(zonal_mean_max_size=LOG_ZONAL_MEAN_IMAGES),
                TimeMeanMetricConfig(target="denorm"),
                TimeMeanMetricConfig(target="norm"),
            ],
        ).build(
            dataset_info=ds_info,
            n_ic_steps=1,
            n_forward_steps=1,
            initial_time=get_zero_time(shape=[1, 0], dims=["sample", "time"]),
            normalize=lambda x: dict(x),
            save_diagnostics=True,
            output_dir=None,
        )


class TestAggregatorConfigMetrics:
    def test_duplicate_metric_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate metric names"):
            InferenceEvaluatorAggregatorConfig(
                metrics=[
                    MeanMetricConfig(target="denorm"),
                    MeanMetricConfig(target="denorm"),
                ]
            )

    def test_duplicate_names_allowed_with_explicit_name(self):
        InferenceEvaluatorAggregatorConfig(
            metrics=[
                MeanMetricConfig(target="denorm"),
                MeanMetricConfig(target="norm", name="mean_custom"),
            ]
        )

    def test_default_metrics_build(self):
        n_sample, n_time = 10, 22
        nx, ny = 90, 45
        ds_info = get_ds_info(nx, ny)
        initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])
        agg = InferenceEvaluatorAggregatorConfig().build(
            dataset_info=ds_info,
            n_ic_steps=1,
            n_forward_steps=n_time - 1,
            initial_time=initial_time,
            normalize=lambda x: dict(x),
            save_diagnostics=False,
        )
        logs = agg.record_batch(
            data=PairedData.new_on_device(
                prediction={
                    "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
                },
                reference={
                    "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
                },
                time=xr.DataArray(
                    np.zeros((n_sample, n_time)), dims=["sample", "time"]
                ),
                labels=None,
            ),
        )
        assert len(logs) == n_time
        assert "mean/weighted_rmse/a" in logs[0]

    def test_explicit_metrics_build(self):
        n_sample, n_time = 10, 22
        nx, ny = 90, 45
        ds_info = get_ds_info(nx, ny)
        initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])
        agg = InferenceEvaluatorAggregatorConfig(
            metrics=[
                MeanMetricConfig(target="denorm"),
                StepMeanMetricConfig(step=20, target="denorm"),
            ]
        ).build(
            dataset_info=ds_info,
            n_ic_steps=1,
            n_forward_steps=n_time - 1,
            initial_time=initial_time,
            normalize=lambda x: dict(x),
            save_diagnostics=False,
        )
        logs = agg.record_batch(
            data=PairedData.new_on_device(
                prediction={
                    "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
                },
                reference={
                    "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
                },
                time=xr.DataArray(
                    np.zeros((n_sample, n_time)), dims=["sample", "time"]
                ),
                labels=None,
            ),
        )
        assert len(logs) == n_time
        assert "mean/weighted_rmse/a" in logs[0]
        summary = agg.get_summary_logs()
        assert "mean_step_20/weighted_rmse/a" in summary

    def test_enable_time_series_false(self):
        n_sample, n_time = 10, 22
        nx, ny = 4, 4
        ds_info = get_ds_info(nx, ny)
        initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])
        agg = InferenceEvaluatorAggregatorConfig().build(
            dataset_info=ds_info,
            n_ic_steps=1,
            n_forward_steps=n_time - 1,
            initial_time=initial_time,
            normalize=lambda x: dict(x),
            save_diagnostics=False,
            enable_time_series=False,
        )
        logs = agg.record_batch(
            data=PairedData.new_on_device(
                prediction={
                    "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
                },
                reference={
                    "a": torch.randn(n_sample, n_time, ny, nx, device=get_device())
                },
                time=xr.DataArray(
                    np.zeros((n_sample, n_time)), dims=["sample", "time"]
                ),
                labels=None,
            ),
        )
        for log in logs:
            assert "mean/weighted_rmse/a" not in log

    def test_raises_without_output_dir(self):
        ds_info = get_ds_info(nx=2, ny=2)
        with pytest.raises(
            ValueError, match="Output directory must be set to save diagnostics"
        ):
            InferenceEvaluatorAggregatorConfig().build(
                dataset_info=ds_info,
                n_ic_steps=1,
                n_forward_steps=1,
                initial_time=get_zero_time(shape=[1, 0], dims=["sample", "time"]),
                normalize=lambda x: dict(x),
                save_diagnostics=True,
                output_dir=None,
            )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_config_matches_typed_config():
    """Verify LegacyFlagInferenceEvaluatorAggregatorConfig produces the same
    summary logs as an equivalent InferenceEvaluatorAggregatorConfig built
    with the same typed metrics."""
    torch.manual_seed(42)
    n_sample, n_time, nx, ny = 4, 22, 90, 45
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    legacy_config = LegacyFlagInferenceEvaluatorAggregatorConfig(
        log_zonal_mean_images=LOG_ZONAL_MEAN_IMAGES,
        log_step_means=[
            StepMeanEntry(step=20),
            StepMeanEntry(step=4, name="one_day_mean"),
        ],
        log_video=True,
    )

    typed_config = InferenceEvaluatorAggregatorConfig(
        metrics=[
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, name="mean_step_20_norm", target="norm"),
            StepMeanMetricConfig(step=4, name="one_day_mean", target="denorm"),
            StepMeanMetricConfig(step=4, name="one_day_mean_norm", target="norm"),
            PowerSpectrumMetricConfig(),
            ZonalMeanMetricConfig(zonal_mean_max_size=LOG_ZONAL_MEAN_IMAGES),
            VideoMetricConfig(),
            TimeMeanMetricConfig(target="denorm"),
            TimeMeanMetricConfig(target="norm"),
        ],
    )

    build_kwargs = dict(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    legacy_agg = legacy_config.build(**build_kwargs)
    typed_agg = typed_config.build(**build_kwargs)

    data = PairedData.new_on_device(
        prediction={"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())},
        reference={"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())},
        time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
        labels=None,
    )
    legacy_agg.record_batch(data=data)
    typed_agg.record_batch(data=data)

    legacy_logs = legacy_agg.get_summary_logs()
    typed_logs = typed_agg.get_summary_logs()

    assert set(legacy_logs.keys()) == set(typed_logs.keys()), (
        f"Key mismatch.\n"
        f"  Only in legacy: {set(legacy_logs) - set(typed_logs)}\n"
        f"  Only in typed:  {set(typed_logs) - set(legacy_logs)}"
    )
    for key in legacy_logs:
        legacy_val = legacy_logs[key]
        typed_val = typed_logs[key]
        if isinstance(legacy_val, int | float):
            torch.testing.assert_close(
                torch.tensor(legacy_val),
                torch.tensor(typed_val),
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch at key {key}",
            )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_config_long_run_includes_annual_and_enso():
    """Verify that a long-run legacy config produces annual and ENSO metrics."""
    torch.manual_seed(42)
    n_sample = 2
    nx, ny = 72, 36
    timestep = datetime.timedelta(days=10)
    n_time = 80
    ds_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lon=torch.linspace(0, 355, nx),
            lat=torch.linspace(-87.5, 87.5, ny),
        ),
        timestep=timestep,
    )
    time_1d = xr.cftime_range(start="2000-01-01", periods=n_time, freq="10D")
    time_values = np.array([time_1d.values] * n_sample)
    time = xr.DataArray(time_values, dims=["sample", "time"])
    initial_time = time.isel(time=0)

    legacy_config = LegacyFlagInferenceEvaluatorAggregatorConfig(
        log_video=True,
        log_seasonal_means=True,
    )

    typed_config = InferenceEvaluatorAggregatorConfig(
        metrics=[
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
            StepMeanMetricConfig(step=20, target="denorm"),
            StepMeanMetricConfig(step=20, name="mean_step_20_norm", target="norm"),
            PowerSpectrumMetricConfig(),
            ZonalMeanMetricConfig(),
            VideoMetricConfig(),
            TimeMeanMetricConfig(target="denorm"),
            TimeMeanMetricConfig(target="norm"),
            SeasonalMetricConfig(),
            AnnualMetricConfig(),
            EnsoIndexMetricConfig(),
        ],
    )

    build_kwargs = dict(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    legacy_agg = legacy_config.build(**build_kwargs)
    typed_agg = typed_config.build(**build_kwargs)

    data = PairedData.new_on_device(
        prediction={"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())},
        reference={"a": torch.randn(n_sample, n_time, ny, nx, device=get_device())},
        time=time,
        labels=None,
    )
    legacy_agg.record_batch(data=data)
    typed_agg.record_batch(data=data)

    legacy_logs = legacy_agg.get_summary_logs()
    typed_logs = typed_agg.get_summary_logs()

    assert set(legacy_logs.keys()) == set(typed_logs.keys()), (
        f"Key mismatch.\n"
        f"  Only in legacy: {set(legacy_logs) - set(typed_logs)}\n"
        f"  Only in typed:  {set(typed_logs) - set(legacy_logs)}"
    )
    annual_keys = [k for k in legacy_logs if "annual" in k]
    assert len(annual_keys) > 0, "Expected annual metric keys in long-run config"
    assert (
        "enso_index" in legacy_agg._aggregators
    ), "Expected enso_index aggregator in long-run config"


def test_hierarchical_defaults_build():
    """Hierarchical config with all defaults builds and includes core metrics."""
    n_sample = 2
    n_time = 24
    nx = 90
    ny = 45
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    agg = HierarchicalInferenceEvaluatorAggregatorConfig(
        enso_coefficient=EnsoCoefficientMetricConfig(enabled=False),
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    for expected in ["mean", "mean_norm", "power_spectrum", "zonal_mean", "time_mean"]:
        assert expected in agg._aggregators, f"Expected {expected} in aggregators"


def test_hierarchical_enable_histogram():
    """Enabling histogram in hierarchical config adds it to the aggregators."""
    n_sample = 2
    n_time = 24
    nx = 90
    ny = 45
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    agg = HierarchicalInferenceEvaluatorAggregatorConfig(
        histogram=HistogramMetricConfig(enabled=True),
        enso_coefficient=EnsoCoefficientMetricConfig(enabled=False),
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    assert "histogram" in agg._aggregators
    assert "mean" in agg._aggregators


def test_hierarchical_disable_default():
    """Disabling a default metric in hierarchical config removes it."""
    n_sample = 2
    n_time = 24
    nx = 90
    ny = 45
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    agg = HierarchicalInferenceEvaluatorAggregatorConfig(
        zonal_mean=ZonalMeanMetricConfig(enabled=False),
        enso_coefficient=EnsoCoefficientMetricConfig(enabled=False),
    ).build(
        dataset_info=ds_info,
        n_ic_steps=1,
        n_forward_steps=n_time - 1,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    assert "zonal_mean" not in agg._aggregators
    assert "mean" in agg._aggregators


@pytest.mark.parametrize(
    "kwargs,match",
    [
        pytest.param(
            dict(mean_denorm=MeanMetricConfig(target="norm")),
            "mean_denorm.target must be 'denorm'",
            id="mean_denorm_wrong_target",
        ),
        pytest.param(
            dict(mean_norm=MeanMetricConfig(target="denorm")),
            "mean_norm.target must be 'norm'",
            id="mean_norm_wrong_target",
        ),
        pytest.param(
            dict(time_mean_denorm=TimeMeanMetricConfig(target="norm")),
            "time_mean_denorm.target must be 'denorm'",
            id="time_mean_denorm_wrong_target",
        ),
        pytest.param(
            dict(time_mean_norm=TimeMeanMetricConfig(target="denorm")),
            "time_mean_norm.target must be 'norm'",
            id="time_mean_norm_wrong_target",
        ),
    ],
)
def test_hierarchical_rejects_mismatched_target(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HierarchicalInferenceEvaluatorAggregatorConfig(**kwargs)


def test_default_aggregator_config_yaml():
    """Regression test ensuring the default aggregator config YAML stays in sync."""
    import dataclasses

    import yaml

    from fme.core.testing.regression import validate_text

    config = HierarchicalInferenceEvaluatorAggregatorConfig()
    content = yaml.dump(
        {"aggregator": dataclasses.asdict(config)},
        default_flow_style=False,
        sort_keys=False,
    )
    docs_path = pathlib.Path(__file__).parents[4] / "docs"
    validate_text(content, docs_path / "default-aggregator-config.yaml")


def test_all_metric_configs_documented():
    """Every type in the MetricConfig union must appear in evaluator_config.rst."""
    import fme.ace
    from fme.ace.aggregator.inference.main import MetricConfig

    docs_path = pathlib.Path(__file__).parents[4] / "docs" / "evaluator_config.rst"
    docs_content = docs_path.read_text()

    for cls in typing.get_args(MetricConfig):
        name = cls.__name__
        assert hasattr(
            fme.ace, name
        ), f"{name} is in MetricConfig union but not exported from fme.ace"
        assert f"fme.ace.{name}" in docs_content, (
            f"{name} is in MetricConfig union but not documented in "
            f"docs/evaluator_config.rst"
        )
