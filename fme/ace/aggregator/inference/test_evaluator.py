import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.inference import InferenceEvaluatorAggregator
from fme.ace.data_loading.batch_data import BatchData, PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device

TIMESTEP = datetime.timedelta(hours=6)


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


def test_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    ds_info = get_ds_info(nx, ny)
    initial_time = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_timesteps=n_time,
        initial_time=initial_time,
        record_step_20=True,
        log_video=True,
        log_zonal_mean_images=True,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    time = xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"])

    logs = agg.record_batch(
        data=PairedData(
            prediction={
                "a": torch.randn(n_sample, n_time, nx, ny, device=get_device())
            },
            reference={"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())},
            time=time,
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
    expected_keys = [
        "mean_step_20/loss",
        "mean_step_20/weighted_rmse/a",
        "mean_step_20/weighted_bias/a",
        "mean_step_20/weighted_grad_mag_percent_diff/a",
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
        "video/a",
    ]
    assert set(summary_logs.keys()) == set(expected_keys)


def test_inference_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    ds_info = get_ds_info(nx, ny)
    initial_time = (get_zero_time(shape=[n_sample, 0], dims=["sample", "time"]),)
    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_timesteps=n_time,
        initial_time=initial_time,
        record_step_20=True,
        log_video=True,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    logs = agg.record_batch(
        data=PairedData(
            prediction={
                "a": torch.randn(n_sample, n_time, nx, ny, device=get_device())
            },
            reference={"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())},
            time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
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
    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_timesteps=window_len * n_windows,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        save_diagnostics=False,
    )
    target_data = BatchData.new_on_device(
        data={"a": torch.zeros([2, window_len, ny, nx], device=get_device())},
        time=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
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
    agg = InferenceEvaluatorAggregator(
        dataset_info=ds_info,
        n_timesteps=n_time,
        initial_time=initial_time,
        normalize=lambda x: dict(x),
        output_dir=tmpdir,
        record_step_20=True,
        log_zonal_mean_images=True,
        log_video=True,
        log_histograms=True,
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    agg.record_batch(
        data=PairedData(
            prediction=gen_data,
            reference=target_data,
            time=time,
        ),
    )
    agg.flush_diagnostics()
    expected_files = [  # note: time-dependent aggregators not tested here
        "mean",
        "mean_norm",
        "mean_step_20",
        "zonal_mean",
        "time_mean",
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
        InferenceEvaluatorAggregator(
            dataset_info=ds_info,
            n_timesteps=1,
            initial_time=get_zero_time(shape=[1, 0], dims=["sample", "time"]),
            normalize=lambda x: dict(x),
            save_diagnostics=True,
            output_dir=None,
        )
