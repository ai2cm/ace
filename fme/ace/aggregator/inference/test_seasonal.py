import datetime

import cftime
import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.aggregator.inference.seasonal import SeasonalAggregator
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.mask_provider import MaskProvider


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ns]"), dims=dims)


def test_seasonal_aggregator():
    n_lat = 16
    n_lon = 32
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_time_step = 8
    n_time = int(365 / 10 * 2 / n_time_step + 1) * n_time_step
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    agg = SeasonalAggregator(
        LatLonOperations(area_weights),
    )
    target_data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())
    }
    gen_data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}

    def time_select(tensor_mapping, start, stop):
        return {
            name: value[:, start:stop, ...] for name, value in tensor_mapping.items()
        }

    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    time_1d = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + i * datetime.timedelta(days=10)
        for i in range(n_time)
    ]
    time = xr.DataArray([time_1d for _ in range(n_sample)], dims=["sample", "time"])
    for i in range(0, n_time, n_time_step):
        agg.record_batch(
            time.isel(time=range(i, i + n_time_step)),
            time_select(target_data, i, i + n_time_step),
            time_select(gen_data, i, i + n_time_step),
        )
    logs = agg.get_logs(label="test")
    for name, value in logs.items():
        if isinstance(value, float | np.ndarray):
            assert not np.isnan(value), f"{name} is nan"


def test_seasonal_aggregator_with_nans():
    n_lat = 16
    n_lon = 32
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_time_step = 8
    n_time = int(365 / 10 * 2 / n_time_step + 1) * n_time_step
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    mask = torch.ones((n_lat, n_lon))
    mask[1, 1] = 0
    mask_provider = MaskProvider({"mask_a": mask}).to(get_device())
    agg = SeasonalAggregator(
        LatLonOperations(area_weights, mask_provider),
    )
    target_data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())
    }
    gen_data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
    target_data["a"][:, :, 1, 1] = float("nan")
    gen_data["a"][:, :, 1, 1] = float("nan")

    def time_select(tensor_mapping, start, stop):
        return {
            name: value[:, start:stop, ...] for name, value in tensor_mapping.items()
        }

    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    time_1d = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + i * datetime.timedelta(days=10)
        for i in range(n_time)
    ]
    time = xr.DataArray([time_1d for _ in range(n_sample)], dims=["sample", "time"])
    for i in range(0, n_time, n_time_step):
        agg.record_batch(
            time.isel(time=range(i, i + n_time_step)),
            time_select(target_data, i, i + n_time_step),
            time_select(gen_data, i, i + n_time_step),
        )
    logs = agg.get_logs(label="test")
    for name, value in logs.items():
        if isinstance(value, float | np.ndarray):
            assert not np.isnan(value), f"{name} is nan"
