import datetime
import pathlib

import cftime
import matplotlib.pyplot as plt
import pytest
import torch
import xarray as xr

import fme
from fme.ace.aggregator.inference.annual import GlobalMeanAnnualAggregator
from fme.ace.testing import DimSizes, MonthlyReferenceData
from fme.core.coordinates import DimSize
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.testing import mock_distributed

TIMESTEP = datetime.timedelta(hours=6)


def test_annual_aggregator(tmpdir):
    n_lat = 16
    n_lon = 32
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_time = 365 * 4 * 2
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    names = ["a"]
    horizontal = [DimSize("grid_yt", n_lat), DimSize("grid_xt", n_lon)]
    monthly_reference_data = MonthlyReferenceData(
        path=pathlib.Path(tmpdir),
        names=names,
        dim_sizes=DimSizes(
            n_time=48,
            horizontal=horizontal,
            nz_interface=1,
        ),
        n_ensemble=3,
    )
    monthly_ds = xr.open_dataset(monthly_reference_data.data_filename)
    agg = GlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights),
        timestep=TIMESTEP,
        monthly_reference_data=monthly_ds,
    )
    target_data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())
    }
    gen_data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
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
    agg.record_batch(time, target_data, gen_data)
    logs = agg.get_logs(label="test")
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)
    assert "test/r2/a_target" in logs


@pytest.mark.parametrize("use_mock_distributed", [False, True])
def test__get_gathered_means(use_mock_distributed):
    """Test the private _get_gathered_means method, rather than the public
    get_logs method, because the public method returns some data converted to images,
    and so that portion can't easily be tested.
    """
    n_lat = 16
    n_lon = 32
    n_sample = 2
    n_time = 365 * 4 * 2  # two years, approximately
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    agg = GlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights),
        timestep=TIMESTEP,
    )
    target_data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())
    }
    gen_data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
    time = xr.DataArray(
        [
            [
                (
                    cftime.DatetimeProlepticGregorian(
                        2000 + j, 1, 1
                    )  # add j so that years are not the same
                    + i * datetime.timedelta(hours=6)
                )
                for i in range(n_time)
            ]
            for j in range(n_sample)
        ],
        dims=["sample", "time"],
    )
    agg.record_batch(time, target_data, gen_data)
    if use_mock_distributed:
        world_size = 2
        with mock_distributed(world_size=world_size):
            result = agg._get_gathered_means()
            assert result is not None
            target, gen = result
            combined = agg.get_dataset()
    else:
        world_size = 1
        result = agg._get_gathered_means()
        assert result is not None
        target, gen = result
        combined = agg.get_dataset()
    for dataset in (target, gen, combined):
        assert set(dataset.dims).issuperset({"sample", "year"})
        assert list(dataset.year.values) == [2000, 2001, 2002]
        assert dataset.sizes["sample"] == n_sample * world_size
    assert set(combined.coords["source"].values) == set(["target", "prediction"])
