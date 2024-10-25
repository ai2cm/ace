import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.aggregator.inference import InferenceEvaluatorAggregator
from fme.core.data_loading.batch_data import BatchData
from fme.core.data_loading.data_typing import LatLonCoordinates, SigmaCoordinates
from fme.core.device import get_device

TIMESTEP = datetime.timedelta(hours=6)


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ns]"), dims=dims)


def test_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    nz = 3
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    horizontal_coordinates = LatLonCoordinates(
        lon=torch.arange(nx),
        lat=torch.arange(ny),
        loaded_lon_name="lon",
        loaded_lat_name="lat",
    )
    initial_times = get_zero_time(shape=[n_sample, 0], dims=["sample", "time"])

    agg = InferenceEvaluatorAggregator(
        sigma_coordinates,
        horizontal_coordinates=horizontal_coordinates,
        timestep=TIMESTEP,
        n_timesteps=n_time,
        initial_times=initial_times,
        record_step_20=True,
        log_video=True,
        log_zonal_mean_images=True,
    )
    times = xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"])
    target_data = BatchData(
        data={"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())},
        times=times,
    )
    gen_data = BatchData(
        data={"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())},
        times=times,
    )

    agg.record_batch(
        prediction=gen_data, target=target_data, normalize=lambda x: x, i_time_start=0
    )
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
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    horizontal_coordinates = LatLonCoordinates(
        lon=torch.arange(nx),
        lat=torch.arange(ny),
        loaded_lon_name="lon",
        loaded_lat_name="lat",
    )
    initial_times = (get_zero_time(shape=[n_sample, 0], dims=["sample", "time"]),)
    agg = InferenceEvaluatorAggregator(
        sigma_coordinates,
        horizontal_coordinates,
        TIMESTEP,
        n_time,
        initial_times,
        record_step_20=True,
        log_video=True,
    )
    target_data = BatchData(
        data={"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())},
        times=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
    )
    gen_data = BatchData(
        data={"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())},
        times=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
    )
    agg.record_batch(
        prediction=gen_data, target=target_data, normalize=lambda x: x, i_time_start=0
    )
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
    nz = 3
    nx, ny = 4, 4
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    horizontal_coordinates = LatLonCoordinates(
        lon=torch.arange(nx),
        lat=torch.arange(ny),
        loaded_lon_name="lon",
        loaded_lat_name="lat",
    )
    initial_times = (get_zero_time(shape=[2, 0], dims=["sample", "time"]),)
    agg = InferenceEvaluatorAggregator(
        sigma_coordinates,
        horizontal_coordinates,
        TIMESTEP,
        (window_len - overlap) * n_windows + 1,
        initial_times,
    )
    target_data = BatchData(
        data={"a": torch.zeros([2, window_len, ny, nx], device=get_device())},
        times=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
    )
    i_start = 0
    for i in range(n_windows):
        sample_data = {"a": torch.zeros([2, window_len, ny, nx], device=get_device())}
        for i in range(window_len):
            sample_data["a"][..., i, :, :] = float(i_start + i)
        predicted = BatchData(
            data=sample_data,
            times=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
        )
        agg.record_batch(
            prediction=predicted,
            target=target_data,
            normalize=lambda x: x,
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
    nz = 3
    nx, ny = 4, 4
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    horizontal_coordinates = LatLonCoordinates(
        lon=torch.arange(nx),
        lat=torch.arange(ny),
        loaded_lon_name="lon",
        loaded_lat_name="lat",
    )
    initial_times = (get_zero_time(shape=[2, 0], dims=["sample", "time"]),)
    agg = InferenceEvaluatorAggregator(
        sigma_coordinates,
        horizontal_coordinates,
        TIMESTEP,
        (window_len - overlap) * n_windows + overlap,
        initial_times,
    )
    target_data = BatchData(
        data={"a": torch.zeros([2, window_len, ny, nx], device=get_device())},
        times=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
    )
    i_start = 0
    for i in range(n_windows):
        sample_data = {"a": torch.zeros([2, window_len, ny, nx], device=get_device())}
        for i in range(window_len):
            sample_data["a"][..., i, :, :] = float(i_start + i)
        predicted = BatchData(
            data=sample_data,
            times=xr.DataArray(np.zeros((2, window_len)), dims=["sample", "time"]),
        )
        agg.record_batch(
            prediction=predicted,
            target=target_data,
            normalize=lambda x: x,
            i_time_start=i_start,
        )
        i_start += window_len - overlap  # subtract 1 for overlapping windows
    logs = agg.get_inference_logs(label="metrics")
    assert len(logs) == (window_len - overlap) * n_windows + overlap
