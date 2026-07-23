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
    get_crps,
    get_r2,
    get_rmse,
)
from fme.ace.aggregator.inference.data import InferenceBatchData
from fme.ace.testing import DimSizes, MonthlyReferenceData
from fme.core.coordinates import DimSize
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.testing import mock_distributed

TIMESTEP = datetime.timedelta(hours=6)


def _record_paired_annual_aggregator(
    tmpdir, report_crps: bool = True, report_rmse: bool = True
) -> PairedGlobalMeanAnnualAggregator:
    """Build a paired annual aggregator and record one batch of two full years
    of random target/prediction data (enough for the plotting and metric paths
    to be exercised)."""
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
        report_crps=report_crps,
        report_rmse=report_rmse,
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
    batch = InferenceBatchData(
        prediction=gen_data,
        prediction_norm={},
        target=target_data,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
    return agg


def test_paired_annual_aggregator(tmpdir):
    agg = _record_paired_annual_aggregator(tmpdir)
    logs = agg.get_logs(label="test")
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)
    assert "test/r2/a_target" in logs
    assert "test/rmse/a" in logs
    assert "test/crps/a" in logs

    # the reported RMSE should equal the RMSE between the ensemble-mean annual
    # evolution of the prediction and that of the target
    ds = agg.get_dataset()
    target_ensemble_mean = ds["a"].sel(source="target").mean("sample")
    gen_ensemble_mean = ds["a"].sel(source="prediction").mean("sample")
    expected_rmse = float(
        np.sqrt(np.nanmean((gen_ensemble_mean - target_ensemble_mean).values ** 2))
    )
    np.testing.assert_allclose(logs["test/rmse/a"], expected_rmse, rtol=1e-5)

    # the reported CRPS should be the fair CRPS of the predicted ensemble annual
    # evolution against the target ensemble-mean annual evolution
    expected_crps = get_crps(ds["a"].sel(source="prediction"), target_ensemble_mean)
    assert not np.isnan(logs["test/crps/a"])
    np.testing.assert_allclose(logs["test/crps/a"], expected_crps, rtol=1e-5)


@pytest.mark.parametrize(
    "report_crps,report_rmse",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_paired_annual_aggregator_metric_toggles(tmpdir, report_crps, report_rmse):
    agg = _record_paired_annual_aggregator(
        tmpdir, report_crps=report_crps, report_rmse=report_rmse
    )
    logs = agg.get_logs(label="test")
    assert ("test/crps/a" in logs) == report_crps
    assert ("test/rmse/a" in logs) == report_rmse
    # the annual-evolution plot and R2 are reported regardless of the toggles
    assert "test/a" in logs
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
    spatial_mask_provider = SpatialMaskProvider({"mask_a": torch.tensor(mask)}).to(
        get_device()
    )
    agg = PairedGlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights, spatial_mask_provider),
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
    batch = InferenceBatchData(
        prediction=gen_data,
        prediction_norm={},
        target=target_data,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
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
    assert not np.isnan(logs["test/rmse/a"])
    assert not np.isnan(logs["test/crps/a"])
    for source in ["target", "gen"]:
        r2 = logs[f"test/r2/a_{source}"]
        assert not np.isnan(r2)
        big_r2 = 50  # account for small denominators
        assert r2 > -big_r2  # can be -inf if NaNs improperly handled
        assert r2 < 1


def test_get_rmse_ignores_nan_gap_years():
    # gap years from reindexing show up as NaN; a year is dropped when either
    # series is NaN (2001 only in da, 2004 only in reference), and the RMSE over
    # the remaining years must match a hand-computed value, not propagate NaN.
    years = [2000, 2001, 2002, 2003, 2004]
    da = xr.DataArray(
        [1.0, np.nan, 3.0, 5.0, 2.0], dims=["year"], coords={"year": years}
    )
    reference = xr.DataArray(
        [1.0, 2.0, 2.0, 2.0, np.nan], dims=["year"], coords={"year": years}
    )
    # squared errors over the valid years 2000/2002/2003: 0, 1, 9 -> mean 10/3
    expected = float(np.sqrt(10.0 / 3.0))
    np.testing.assert_allclose(get_rmse(da, reference), expected, rtol=1e-12)


def test_get_r2_ignores_nan_gap_years():
    # get_r2 sees the same NaN gap years as get_rmse and must ignore them
    # rather than returning NaN, dropping a year when either series is NaN
    # (2001 only in da, 2004 only in reference).
    years = [2000, 2001, 2002, 2003, 2004]
    da = xr.DataArray(
        [1.0, np.nan, 3.0, 5.0, 2.0], dims=["year"], coords={"year": years}
    )
    reference = xr.DataArray(
        [1.0, 2.0, 2.0, 2.0, np.nan], dims=["year"], coords={"year": years}
    )
    # over the valid years 2000/2002/2003: ref mean 5/3,
    # SS_ref = (1-5/3)^2 + (2-5/3)^2 + (2-5/3)^2 = 2/3; SS_pred = 0+1+9 = 10
    expected = float(1 - 10.0 / (2.0 / 3.0))
    result = get_r2(da, reference)
    assert not np.isnan(result)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_get_crps_ignores_nan_gap_years():
    # gap years from reindexing show up as NaN in either the prediction or the
    # reference; the CRPS over the remaining years must match a hand-computed
    # value, not propagate NaN.
    years = [2000, 2001, 2002, 2003]
    da = xr.DataArray(
        [[1.0, np.nan, 3.0, 5.0], [2.0, np.nan, 4.0, 6.0]],
        dims=["sample", "year"],
        coords={"year": years},
    )
    reference = xr.DataArray(
        [1.0, 2.0, 2.0, np.nan], dims=["year"], coords={"year": years}
    )
    # fair CRPS per year (2 members) is mean_i|x_i - y| - 0.5 |x_0 - x_1|:
    #   2000: members [1, 2], y=1 -> (0 + 1)/2 - 0.5 * 1 = 0.0
    #   2002: members [3, 4], y=2 -> (1 + 2)/2 - 0.5 * 1 = 1.0
    # 2001 is dropped (NaN prediction) and 2003 is dropped (NaN reference):
    #   mean over the remaining years 2000/2002: (0 + 1)/2 = 0.5
    expected = 0.5
    result = get_crps(da, reference)
    assert not np.isnan(result)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


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
    batch = InferenceBatchData(
        prediction=gen_data,
        prediction_norm={},
        target=target_data,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
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
    batch = InferenceBatchData(
        prediction=data,
        prediction_norm={},
        target=None,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
    logs = agg.get_logs(label="test")
    assert len(logs) > 0
    assert "test/a" in logs
    assert isinstance(logs["test/a"], plt.Figure)


FIVE_DAILY_TIMESTEP = datetime.timedelta(days=5)


def _get_kept_years(
    start: cftime.DatetimeNoLeap,
    timestep: datetime.timedelta,
    n_time: int,
) -> list[int]:
    """Record a single-sample rollout of random data starting at ``start`` and
    return the years retained in the annual mean series."""
    n_lat, n_lon = 2, 2
    agg = GlobalMeanAnnualAggregator(
        ops=LatLonOperations(torch.ones(n_lat, n_lon).to(fme.get_device())),
        timestep=timestep,
    )
    data = {"a": torch.randn(1, n_time, n_lat, n_lon, device=get_device())}
    time = xr.DataArray(
        [[start + i * timestep for i in range(n_time)]],
        dims=["sample", "time"],
    )
    batch = InferenceBatchData(
        prediction=data,
        prediction_norm={},
        target=None,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
    ds = agg.get_gathered_means()
    assert ds is not None
    return list(ds["year"].values)


def test_annual_aggregator_drops_trailing_final_instant_year_five_daily():
    # 73 five-day steps span a noleap year, so sample 146 lands exactly on
    # 2002-01-01: the trailing year holds a single sample and must be dropped.
    years = _get_kept_years(
        cftime.DatetimeNoLeap(2000, 1, 1), FIVE_DAILY_TIMESTEP, n_time=147
    )
    assert years == [2000, 2001]


def test_annual_aggregator_drops_partial_midyear_start_five_daily():
    # a rollout genuinely starting in July covers only ~184 days of its first
    # year, which must be dropped.
    years = _get_kept_years(
        cftime.DatetimeNoLeap(2000, 7, 1), FIVE_DAILY_TIMESTEP, n_time=37 + 73
    )
    assert years == [2001]


def test_annual_aggregator_keeps_nudged_start_year_six_hourly():
    # a rollout starting a few days into the year (e.g. a nudged initial
    # condition) covers 360 of 365 days of its first year, which is kept.
    years = _get_kept_years(
        cftime.DatetimeNoLeap(2000, 1, 6), TIMESTEP, n_time=360 * 4 + 1460
    )
    assert years == [2000, 2001]


def test_annual_aggregator_drops_february_start_year_six_hourly():
    # a start well past the nudging allowance covers only 334 days of its
    # first year, which must still be dropped.
    years = _get_kept_years(
        cftime.DatetimeNoLeap(2000, 2, 1), TIMESTEP, n_time=334 * 4 + 1460
    )
    assert years == [2001]


def test_paired_annual_aggregator_drops_trailing_final_instant_year_five_daily():
    # both target and prediction series must apply the same completeness rule
    n_lat, n_lon = 2, 2
    n_time = 147
    agg = PairedGlobalMeanAnnualAggregator(
        ops=LatLonOperations(torch.ones(n_lat, n_lon).to(fme.get_device())),
        timestep=FIVE_DAILY_TIMESTEP,
    )
    data = {"a": torch.randn(1, n_time, n_lat, n_lon, device=get_device())}
    target = {"a": torch.randn(1, n_time, n_lat, n_lon, device=get_device())}
    time = xr.DataArray(
        [
            [
                cftime.DatetimeNoLeap(2000, 1, 1) + i * FIVE_DAILY_TIMESTEP
                for i in range(n_time)
            ]
        ],
        dims=["sample", "time"],
    )
    batch = InferenceBatchData(
        prediction=data,
        prediction_norm={},
        target=target,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
    ds = agg.get_dataset()
    assert list(ds["year"].values) == [2000, 2001]


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
    spatial_mask_provider = SpatialMaskProvider(masks).to(get_device())

    agg = GlobalMeanAnnualAggregator(
        ops=LatLonOperations(area_weights, spatial_mask_provider), timestep=TIMESTEP
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
    batch = InferenceBatchData(
        prediction=data,
        prediction_norm={},
        target=None,
        target_norm=None,
        time=time,
        i_time_start=0,
    )
    agg.record_batch(batch)
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
