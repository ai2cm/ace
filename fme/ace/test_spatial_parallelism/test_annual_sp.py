import numpy as np
import pytest
import torch
import os

import datetime
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from fme.core.device import get_device
from fme.core.mask_provider import MaskProvider
from fme.ace.aggregator.inference.annual import (
    GlobalMeanAnnualAggregator,
)
from fme.core.coordinates import (
    LatLonCoordinates,
)
from fme.core.distributed import Distributed

TIMESTEP = datetime.timedelta(hours=6)
tmp_path="testdata"
def int():
    # Define the sizes
    torch.manual_seed(42)
    torch.set_printoptions(precision=12, sci_mode=False)  # Adjust precision as needed
    n_sample = 1  # Example batch size
    n_lat = 180  # Example size for latitude
    n_lon = 360  # Example size for longitude
    n_time = 365 * 4 * 2

    # Create the latitude tensor
    lat = torch.linspace(-90, 90, n_lat)

    # Create the longitude tensor
    lon = torch.linspace(0, 360, n_lon)

    input_tensor = torch.randn(n_sample, n_time, n_lat, n_lon)
    torch.save(input_tensor, os.path.join(tmp_path, "input-annual-test.pt"))
    return lat, lon, n_lat, n_lon, n_sample, n_time

def test_annual_aggregator_wo_sp():
    os.environ['H_PARALLEL_SIZE'] = '1'
    os.environ['W_PARALLEL_SIZE'] = '1'
    # need to have two actual full years of data for plotting to get exercised
    lat_host, lon_host, n_lat, n_lon, n_sample, n_time = int()
    input_ = torch.load(os.path.join(tmp_path, "input-annual-test.pt"))

    device=get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())

    agg = GlobalMeanAnnualAggregator(
        ops=gridded_ops, timestep=TIMESTEP
    )
    data = {"a": input_.to(device)}

    time = xr.DataArray(
        [
            [
                (
                    cftime.DatetimeProlepticGregorian(2000, 1, 1)
                    + i * datetime.timedelta(hours=6)
                )
                for i in range(n_time)
            ]
            for _ in range(n_sample)
        ],
        dims=["sample", "time"],
    )
    agg.record_batch(time, data)
    logs = agg.get_logs(label="test")
    print(logs)
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)
    figure=logs["test/a"]
    figure.savefig("test.png")


def test_annual_aggregator_w_sp():
    os.environ['H_PARALLEL_SIZE'] = '2'
    os.environ['W_PARALLEL_SIZE'] = '1'
    # need to have two actual full years of data for plotting to get exercised
    lat_host, lon_host, n_lat, n_lon, n_sample, n_time = int()
    input_ = torch.load(os.path.join(tmp_path, "input-annual-test.pt"))
    dist = Distributed.get_instance()
    device=get_device()
    inp_local_host = (input_[:,:,*dist.get_local_slices((n_lat,n_lon))]).detach().clone()
    inp_local=inp_local_host.to(device)

    device=get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())

    agg = GlobalMeanAnnualAggregator(
        ops=gridded_ops, timestep=TIMESTEP
    )
    data = {"a": inp_local}

    time = xr.DataArray(
        [
            [
                (
                    cftime.DatetimeProlepticGregorian(2000, 1, 1)
                    + i * datetime.timedelta(hours=6)
                )
                for i in range(n_time)
            ]
            for _ in range(n_sample)
        ],
        dims=["sample", "time"],
    )
    agg.record_batch(time, data)
    logs = agg.get_logs(label="test")
    print(logs, device)
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)
    figure=logs["test/a"]
    figure.savefig("test-sp.png")
