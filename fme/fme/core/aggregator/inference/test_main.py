import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.core.aggregator.inference import InferenceAggregator
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ms]"), dims=dims)


def test_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    nz = 3
    loss = 1.0
    area_weights = torch.ones(ny).to(fme.get_device())
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))

    agg = InferenceAggregator(
        area_weights,
        sigma_coordinates,
        record_step_20=True,
        log_video=True,
        log_zonal_mean_images=True,
        n_timesteps=n_time,
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    agg.record_batch(loss, time, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_logs(label="test")
    assert "test/mean/series" in logs
    assert "test/mean_norm/series" in logs
    assert "test/mean_step_20/weighted_rmse/a" in logs
    assert "test/mean_step_20/weighted_bias/a" in logs
    assert "test/mean_step_20/weighted_grad_mag_percent_diff/a" in logs
    table = logs["test/mean/series"]
    assert table.columns == [
        "forecast_step",
        "weighted_bias/a",
        "weighted_grad_mag_percent_diff/a",
        "weighted_mean_gen/a",
        "weighted_mean_target/a",
        "weighted_rmse/a",
        "weighted_std_gen/a",
    ]
    assert "test/time_mean/rmse/a" in logs
    assert "test/time_mean/bias/a" in logs
    assert "test/time_mean/bias_map/a" in logs
    assert "test/time_mean/gen_map/a" in logs
    assert "test/zonal_mean/error/a" in logs
    assert "test/zonal_mean/gen/a" in logs
    assert "test/video/a" in logs


def test_inference_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    nz = 3
    loss = 1.0
    area_weights = torch.ones(ny).to(fme.get_device())
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    agg = InferenceAggregator(
        area_weights,
        sigma_coordinates,
        record_step_20=True,
        log_video=True,
        n_timesteps=n_time,
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    agg.record_batch(loss, time, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_inference_logs(label="test")
    assert isinstance(logs, list)
    assert len(logs) == n_time
    assert "test/mean/weighted_bias/a" in logs[0]
    assert "test/mean/weighted_mean_gen/a" in logs[0]
    assert "test/mean/weighted_mean_target/a" in logs[0]
    assert "test/mean/weighted_grad_mag_percent_diff/a" in logs[0]
    assert "test/mean/weighted_rmse/a" in logs[0]
    assert "test/mean_norm/weighted_bias/a" in logs[0]
    assert "test/mean_norm/weighted_mean_gen/a" in logs[0]
    assert "test/mean_norm/weighted_mean_target/a" in logs[0]
    assert "test/mean_norm/weighted_rmse/a" in logs[0]
    # series/table data should be rolled out, not included as a table
    assert "test/mean/series" not in logs[0]
    assert "test/mean_norm/series" not in logs[0]
    assert "test/reduced/series" not in logs[0]
    assert "test/reduced_norm/series" not in logs[0]


@pytest.mark.parametrize(
    "window_len, n_windows",
    [
        pytest.param(3, 1, id="single_window"),
        pytest.param(3, 2, id="two_windows"),
    ],
)
def test_i_time_start_gets_correct_time_longer_windows(window_len: int, n_windows: int):
    # while this directly tests the "mean" result, this is really a test that
    # the data from the correct timestep is piped into the aggregator.
    overlap = 1  # tested code assumes windows have one overlapping point
    area_weights = torch.ones(4).to(fme.get_device())
    nz = 3
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    agg = InferenceAggregator(
        area_weights,
        sigma_coordinates,
        n_timesteps=(window_len - overlap) * n_windows + 1,
    )
    target_data = {"a": torch.zeros([2, window_len, 4, 4], device=get_device())}
    time = get_zero_time(shape=[2, window_len], dims=["sample", "time"])
    i_start = 0
    for i in range(n_windows):
        sample_data = {"a": torch.zeros([2, window_len, 4, 4], device=get_device())}
        for i in range(window_len):
            sample_data["a"][..., i, :, :] = float(i_start + i)
        agg.record_batch(
            1.0,
            time=time,
            target_data=target_data,
            gen_data=sample_data,
            target_data_norm=target_data,
            gen_data_norm=sample_data,
            i_time_start=i_start,
        )
        i_start += window_len - overlap  # subtract 1 for overlapping windows
    logs = agg.get_logs(label="metrics")
    table = logs["metrics/mean/series"]
    # get the weighted_bias column
    bias = table.get_column("weighted_bias/a")
    assert len(bias) == (window_len - overlap) * n_windows + overlap
    for i in range(len(bias)):
        np.testing.assert_allclose(bias[i], float(i), rtol=1e-5)


@pytest.mark.parametrize(
    "window_len, n_windows, overlap",
    [
        pytest.param(3, 1, 0, id="single_window"),
        pytest.param(3, 2, 0, id="two_windows"),
        pytest.param(3, 2, 1, id="two_windows_overlap"),
    ],
)
def test_inference_logs_length(window_len: int, n_windows: int, overlap: int):
    """
    Test that the inference logs are the correct length when using one or more
    possibly-overlapping windows.
    """
    area_weights = torch.ones(4).to(fme.get_device())
    nz = 3
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    agg = InferenceAggregator(
        area_weights,
        sigma_coordinates,
        n_timesteps=(window_len - overlap) * n_windows + overlap,
    )
    target_data = {"a": torch.zeros([2, window_len, 4, 4], device=get_device())}
    time = get_zero_time(shape=[2, window_len], dims=["sample", "time"])
    i_start = 0
    for i in range(n_windows):
        sample_data = {"a": torch.zeros([2, window_len, 4, 4], device=get_device())}
        for i in range(window_len):
            sample_data["a"][..., i, :, :] = float(i_start + i)
        agg.record_batch(
            1.0,
            time=time,
            target_data=target_data,
            gen_data=sample_data,
            target_data_norm=target_data,
            gen_data_norm=sample_data,
            i_time_start=i_start,
        )
        i_start += window_len - overlap  # subtract 1 for overlapping windows
    logs = agg.get_inference_logs(label="metrics")
    assert len(logs) == (window_len - overlap) * n_windows + overlap
