import datetime
import pathlib

import cftime
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.aggregator.inference.annual import (
    GlobalMeanAnnualAggregator,
    PairedGlobalMeanAnnualAggregator,
)
from fme.ace.testing import DimSizes, MonthlyReferenceData
from fme.core.coordinates import DimSize
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.mask_provider import MaskProvider
from fme.core.testing import mock_distributed

TIMESTEP = datetime.timedelta(hours=6)


def test_paired_annual_aggregator(tmpdir):
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
    monthly_ds = xr.open_dataset(
        monthly_reference_data.data_filename,
        decode_timedelta=False,
    )
    agg = PairedGlobalMeanAnnualAggregator(
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


def test_paired_annual_aggregator_with_nans(tmpdir):
    torch.manual_seed(0)
    n_lat = 16
    n_lon = 32
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_years = 5
    n_time = 365 * 4 * n_years
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    names = ["a"]
    horizontal = [DimSize("grid_yt", n_lat), DimSize("grid_xt", n_lon)]
    monthly_reference_data = MonthlyReferenceData(
        path=pathlib.Path(tmpdir),
        names=names,
        dim_sizes=DimSizes(
            n_time=n_years * 12,
            horizontal=horizontal,
            nz_interface=1,
        ),
        n_ensemble=3,
    )
    monthly_ds = xr.open_dataset(
        monthly_reference_data.data_filename,
        decode_timedelta=False,
    )
    mask = np.ones((n_lat, n_lon))
    mask[1, 1] = 0
    monthly_ds["a"] = monthly_ds["a"].where(mask > 0)
    mask_provider = MaskProvider({"mask_a": torch.tensor(mask)}).to(get_device())
    agg = PairedGlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights, mask_provider),
        timestep=TIMESTEP,
        monthly_reference_data=monthly_ds,
    )
    target_data = {
        "a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())
    }
    gen_data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
    target_data["a"][:, :, 1, 1] = float("nan")
    gen_data["a"][:, :, 1, 1] = float("nan")
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
    annual_mean_series = agg.get_dataset()["a"].values
    expected_series = (
        xr.DataArray(
            np.concatenate(
                [
                    target_data["a"].unsqueeze(dim=0).cpu().numpy(),
                    gen_data["a"].unsqueeze(dim=0).cpu().numpy(),
                ]
            ),
            dims={
                "source": ["target", "prediction"],
                "sample": np.arange(n_sample),
                "time": time,
                "lat": np.arange(n_lat),
                "lon": np.arange(n_lon),
            },
        )
        .groupby(time.isel(sample=0).dt.year)
        .mean(dim=["time", "lat", "lon"])
    ).values
    np.testing.assert_allclose(
        annual_mean_series, expected_series, rtol=1e-4, atol=1e-7
    )
    logs = agg.get_logs(label="test")
    for source in ["target", "gen"]:
        r2 = logs[f"test/r2/a_{source}"]
        assert not np.isnan(r2)
        big_r2 = 50  # account for small denominators
        assert r2 > -big_r2  # can be -inf if NaNs improperly handled
        assert r2 < 1


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
    agg = PairedGlobalMeanAnnualAggregator(
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


def test_annual_aggregator():
    n_lat = 4
    n_lon = 8
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_time = 365 * 4 * 2
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())
    agg = GlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights), timestep=TIMESTEP
    )
    data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
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
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)


def test_annual_aggregator_with_nans():
    torch.manual_seed(0)
    n_lat = 4
    n_lon = 8
    # need to have two actual full years of data for plotting to get exercised
    n_sample = 2
    n_time = 365 * 4 * 2
    area_weights = torch.ones(n_lat, n_lon).to(fme.get_device())

    data = {"a": torch.randn(n_sample, n_time, n_lat, n_lon, device=get_device())}
    data["a"][:, :, 1, 1] = float("nan")
    masks = {"mask_a": torch.ones_like(data["a"][0, 0])}
    masks["mask_a"][1, 1] = 0
    mask_provider = MaskProvider(masks).to(get_device())

    agg = GlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights, mask_provider), timestep=TIMESTEP
    )
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
    expected_series = (
        xr.DataArray(
            data["a"].cpu().numpy(),
            dims={
                "sample": np.arange(n_sample),
                "time": time,
                "lat": np.arange(n_lat),
                "lon": np.arange(n_lon),
            },
        )
        .groupby(time.isel(sample=0).dt.year)
        .mean(dim=["time", "lat", "lon"])
    ).values
    annual_mean_series = agg.get_dataset()["a"].values
    np.testing.assert_allclose(
        annual_mean_series, expected_series, rtol=1e-4, atol=1e-7
    )
