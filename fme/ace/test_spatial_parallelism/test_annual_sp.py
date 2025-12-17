import datetime
import os

import cftime
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.inference.annual import GlobalMeanAnnualAggregator
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.mask_provider import MaskProvider

TIMESTEP = datetime.timedelta(hours=6)
tmp_path = "testdata"


def int():
    # Define the sizes
    torch.manual_seed(42)
    torch.set_printoptions(precision=12, sci_mode=False)  # Adjust precision as needed
    n_sample = 4  # Example batch size
    n_lat = 18  # Example size for latitude
    n_lon = 36  # Example size for longitude
    n_time = 365 * 4 * 30

    # Create the latitude tensor
    lat = torch.linspace(-90, 90, n_lat)

    # Create the longitude tensor
    lon = torch.linspace(0, 360, n_lon)

    input_tensor = torch.randn(n_sample, n_time, n_lat, n_lon)
    input_tensor.is_shared_mp = ["spatial"]
    # torch.save(input_tensor, os.path.join(tmp_path, "input-annual-test.pt"))
    return lat, lon, n_lat, n_lon, n_sample, n_time, input_tensor


def test_annual_aggregator_wo_sp():
    os.environ["H_PARALLEL_SIZE"] = "1"
    os.environ["W_PARALLEL_SIZE"] = "1"
    # need to have two actual full years of data for plotting to get exercised
    lat_host, lon_host, n_lat, n_lon, n_sample, n_time, input_ = int()
    # input_ = torch.load(os.path.join(tmp_path, "input-annual-test.pt"))

    device = get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())

    agg = GlobalMeanAnnualAggregator(ops=gridded_ops, timestep=TIMESTEP)
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
    figure = logs["test/a"]
    figure.savefig("test.png")
    for ax in figure.get_axes():
        # Loop through each line in the axes
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            np.savetxt(tmp_path + "/y_data.txt", y_data)
            np.savetxt(tmp_path + "/x_data.txt", x_data)
            # print("X data:", x_data)
            # print("Y data:", y_data)


def test_annual_aggregator_w_sp():
    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "1"
    # need to have two actual full years of data for plotting to get exercised
    lat_host, lon_host, n_lat, n_lon, n_sample, n_time, input_ = int()
    dist = Distributed.get_instance()
    device = get_device()
    inp_local_host = (
        (input_[:, :, *dist.get_local_slices((n_lat, n_lon))]).detach().clone()
    )
    inp_local = inp_local_host.to(device)

    device = get_device()
    lat = lat_host.to(device)
    lon = lon_host.to(device)
    coords = LatLonCoordinates(lat=lat, lon=lon)
    gridded_ops = coords.get_gridded_operations(mask_provider=MaskProvider())

    agg = GlobalMeanAnnualAggregator(ops=gridded_ops, timestep=TIMESTEP)
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
    y_data_ref = np.loadtxt(tmp_path + "/y_data.txt")
    x_data_ref = np.loadtxt(tmp_path + "/x_data.txt")
    print(logs)

    if len(logs) > 0:
        # assert len(logs) > 0
        assert "test/a" in logs
        assert isinstance(logs["test/a"], plt.Figure)
        figure = logs["test/a"]
        figure.savefig("test-sp.png")
        for ax in figure.get_axes():
            # Loop through each line in the axes
            for line in ax.get_lines():
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                np.testing.assert_allclose(
                    y_data_ref,
                    y_data,
                    rtol=5e-05,
                    atol=1e-10,
                    equal_nan=False,
                    err_msg="",
                    verbose=True,
                )
                np.testing.assert_allclose(
                    x_data_ref,
                    x_data,
                    rtol=1e-08,
                    atol=1e-13,
                    equal_nan=False,
                    err_msg="",
                    verbose=True,
                )
