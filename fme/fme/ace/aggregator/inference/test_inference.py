import datetime

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.aggregator.inference import InferenceAggregator
from fme.ace.data_loading.batch_data import BatchData
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations

TIMESTEP = datetime.timedelta(hours=6)


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ns]"), dims=dims)


def test_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    area_weights = torch.ones(ny).to(fme.get_device())
    agg = InferenceAggregator(
        LatLonOperations(area_weights),
        n_timesteps=n_time,
    )
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    logs = agg.record_batch(
        BatchData(data=gen_data, time=time),
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
    expected_summary_keys = ["time_mean/gen_map/a"]
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
    area_weights = torch.ones(ny).to(fme.get_device())
    reference_time_means = xr.Dataset(
        {
            "a": xr.DataArray(
                np.random.randn(ny, nx),
                dims=["grid_yt", "grid_xt"],
            )
        }
    )
    agg = InferenceAggregator(
        LatLonOperations(area_weights),
        n_timesteps=n_time,
        time_mean_reference_data=reference_time_means,
    )
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    logs = agg.record_batch(
        BatchData(
            data=gen_data,
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
    ]
    for key in expected_summary_keys:
        assert key in summary_logs, key
    assert len(summary_logs) == len(expected_summary_keys), set(
        summary_logs
    ).difference(expected_summary_keys)
