import datetime
import pathlib

import cftime
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import fme
from fme.core.aggregator.inference.annual import GlobalMeanAnnualAggregator
from fme.core.device import get_device
from fme.core.testing import DimSizes, MonthlyReferenceData


def get_zero_time(shape, dims):
    return xr.DataArray(np.zeros(shape, dtype="datetime64[ms]"), dims=dims)


def test_annual_aggregator(tmpdir):
    n_lat = 16
    n_lon = 32
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_time = 365 * 4 * 2
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    names = ["a"]
    monthly_reference_data = MonthlyReferenceData(
        path=pathlib.Path(tmpdir),
        names=names,
        dim_sizes=DimSizes(
            n_time=48,
            n_lat=n_lat,
            n_lon=n_lon,
            nz_interface=1,
        ),
        n_ensemble=3,
    )
    monthly_ds = xr.open_dataset(monthly_reference_data.data_filename)
    agg = GlobalMeanAnnualAggregator(
        area_weights=area_weights,
        monthly_reference_data=monthly_ds,
    )
    target_data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())
    }
    gen_data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
    time = get_zero_time(shape=[n_sample, n_time], dims=["sample", "time"])
    time_1d = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + i * datetime.timedelta(hours=6)
        for i in range(n_time)
    ]
    time = xr.DataArray([time_1d for _ in range(n_sample)], dims=["sample", "time"])
    agg.record_batch(time, target_data, gen_data)
    logs = agg.get_logs(label="test")
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)
    assert "test/r2/a_target" in logs
