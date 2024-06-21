import datetime

import numpy as np
import torch
import xarray as xr

import fme
from fme.core.aggregator.inference import InferenceAggregator
from fme.core.data_loading.data_typing import SigmaCoordinates
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
    area_weights = torch.ones(ny).to(fme.get_device())
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))

    agg = InferenceAggregator(
        area_weights,
        sigma_coordinates,
        TIMESTEP,
    )
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    agg.record_batch(time, gen_data)
    logs = agg.get_logs(label="test")
    assert "test/time_mean/gen_map/a" in logs


def test_inference_logs_labels_exist():
    n_sample = 10
    n_time = 22
    nx = 2
    ny = 2
    nz = 3
    area_weights = torch.ones(ny).to(fme.get_device())
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    agg = InferenceAggregator(
        area_weights,
        sigma_coordinates,
        TIMESTEP,
    )
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    agg.record_batch(time, gen_data)
    logs = agg.get_inference_logs(label="test")
    assert isinstance(logs, list)
    assert len(logs) == 1
    # assert len(logs) == n_time use this assertion when timeseries data is generated
    assert "test/time_mean/gen_map/a" in logs[-1]
