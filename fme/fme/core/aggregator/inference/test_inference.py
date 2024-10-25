import datetime

import numpy as np
import torch
import xarray as xr

import fme
from fme.core.aggregator.inference import InferenceAggregator
from fme.core.data_loading.batch_data import BatchData
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
    agg.record_batch(
        BatchData(
            data=gen_data,
            times=time,
        ),
        normalize=None,
        i_time_start=0,
    )
    logs = agg.get_logs(label="test")
    assert "test/mean/series" in logs
    assert "test/time_mean/gen_map/a" in logs
    assert "test/time_mean/ref_bias_map/a" not in logs
    assert "test/time_mean/ref_bias/a" not in logs
    assert "test/time_mean/ref_rmse/a" not in logs


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
    agg.record_batch(
        BatchData(
            data=gen_data,
            times=time,
        ),
        normalize=None,
        i_time_start=0,
    )
    logs = agg.get_logs(label="test")
    assert "test/mean/series" in logs
    assert "test/time_mean/gen_map/a" in logs
    assert "test/time_mean/ref_bias_map/a" in logs
    assert "test/time_mean/ref_bias/a" in logs
    assert "test/time_mean/ref_rmse/a" in logs


def test_inference_logs_labels_exist():
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
    agg.record_batch(
        BatchData(
            data=gen_data,
            times=time,
        ),
        normalize=None,
        i_time_start=0,
    )
    logs = agg.get_inference_logs(label="test")
    assert isinstance(logs, list)
    assert len(logs) == n_time
    assert "test/mean/weighted_mean_gen/a" in logs[0]
    assert "test/mean/weighted_mean_gen/a" in logs[-1]
    # assert len(logs) == n_time use this assertion when timeseries data is generated
    assert "test/time_mean/gen_map/a" in logs[-1]
