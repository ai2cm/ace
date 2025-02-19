import datetime

import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.inference import InferenceAggregator
from fme.ace.data_loading.batch_data import PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device

TIMESTEP = datetime.timedelta(hours=6)


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ns]"), dims=dims)


def test_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    lat, lon = torch.linspace(-90, 90, ny), torch.linspace(-180, 180, nx)
    agg = InferenceAggregator(
        LatLonCoordinates(lat, lon).to(device=get_device()),
        n_time,
        datetime.timedelta(seconds=1),
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    logs = agg.record_batch(
        PairedData(reference=target_data, prediction=gen_data, time=time),
    )
    assert len(logs) == n_time
    expected_step_keys = [
        "mean/forecast_step",
        "mean/weighted_mean_gen/a",
        "mean/weighted_std_gen/a",
    ]
    for log in logs:
        for key in expected_step_keys:
            assert key in log, key
        assert len(log) == len(expected_step_keys), set(log).difference(
            expected_step_keys
        )
    summary_logs = agg.get_summary_logs()
    expected_summary_keys = ["time_mean/gen_map/a", "annual/a"]
    for key in expected_summary_keys:
        assert key in summary_logs, key
    assert len(summary_logs) == len(expected_summary_keys), set(
        summary_logs
    ).difference(expected_summary_keys)


def test_logs_labels_exist_with_reference_time_means():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    lat, lon = torch.linspace(-90, 90, ny), torch.linspace(-180, 180, nx)
    reference_time_means = xr.Dataset(
        {
            "a": xr.DataArray(
                np.random.randn(ny, nx),
                dims=["grid_yt", "grid_xt"],
            )
        }
    )
    agg = InferenceAggregator(
        LatLonCoordinates(lat, lon).to(device=get_device()),
        n_time,
        datetime.timedelta(seconds=1),
        time_mean_reference_data=reference_time_means,
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    logs = agg.record_batch(
        PairedData(
            reference=target_data,
            prediction=gen_data,
            time=time,
        ),
    )
    assert len(logs) == n_time
    expected_step_keys = [
        "mean/forecast_step",
        "mean/weighted_mean_gen/a",
        "mean/weighted_std_gen/a",
    ]
    for log in logs:
        for key in expected_step_keys:
            assert key in log, key
        assert len(log) == len(expected_step_keys), set(log).difference(
            expected_step_keys
        )
    summary_logs = agg.get_summary_logs()
    expected_summary_keys = [
        "time_mean/gen_map/a",
        "time_mean/ref_bias_map/a",
        "time_mean/ref_bias/a",
        "time_mean/ref_rmse/a",
        "annual/a",
    ]
    for key in expected_summary_keys:
        assert key in summary_logs, key
    assert len(summary_logs) == len(expected_summary_keys), set(
        summary_logs
    ).difference(expected_summary_keys)
