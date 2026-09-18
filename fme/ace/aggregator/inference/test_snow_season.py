import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.testing import mock_distributed

from .build_context import MetricBuildContext, MetricNotSupportedError
from .data import InferenceBatchData
from .main import InferenceEvaluatorAggregatorConfig
from .snow_season import (
    QUANTITIES,
    SnowSeasonAggregator,
    SnowSeasonMetricConfig,
    phenology,
    water_year_slices,
)
from .utils import LatLonBoxConfig

LAT = torch.tensor([-45.0, -15.0, 15.0, 45.0])
LON = torch.tensor([0.0, 90.0, 180.0, 270.0])
NORTH = LatLonBoxConfig("north", [0.0, 90.0], [0.0, 360.0])
SOUTH = LatLonBoxConfig("south", [-90.0, 0.0], [0.0, 360.0])
VARIABLE = "swe"
TIMESTEP = datetime.timedelta(days=1)


def _daily_time(
    start: tuple[int, int, int],
    n_time: int,
    n_sample: int = 1,
    calendar: str = "noleap",
) -> xr.DataArray:
    first = cftime.datetime(*start, calendar=calendar)
    times = [first + i * TIMESTEP for i in range(n_time)]
    return xr.DataArray([times for _ in range(n_sample)], dims=["sample", "time"])


def _ops() -> LatLonOperations:
    return LatLonOperations(torch.ones(len(LAT), len(LON)).to(get_device()))


def _aggregator(**kwargs) -> SnowSeasonAggregator:
    defaults = dict(
        ops=_ops(),
        lat=LAT,
        lon=LON,
        variable=VARIABLE,
        regions=[NORTH],
        start_month_northern=10,
        start_month_southern=4,
    )
    defaults.update(kwargs)
    return SnowSeasonAggregator(**defaults)


def triangular_season(time: xr.DataArray, start_month: int, peak: float) -> np.ndarray:
    """Snow that rises linearly from day 30 of the water year to a peak at day 150
    and falls linearly to zero at day 240, shape (sample, time)."""
    out = np.zeros(time.shape)
    for s in range(time.shape[0]):
        dates = list(time.values[s])
        starts = [
            i for i, d in enumerate(dates) if d.month == start_month and d.day == 1
        ]
        for i0 in starts:
            for k in range(365):
                if i0 + k >= len(dates):
                    break
                if 30 <= k <= 150:
                    out[s, i0 + k] = peak * (k - 30) / 120
                elif 150 < k <= 240:
                    out[s, i0 + k] = peak * (240 - k) / 90
    return out


def _fields(series: np.ndarray, rows: slice = slice(None)) -> torch.Tensor:
    """Broadcast a (sample, time) series onto the grid rows given; other rows NaN."""
    n_sample, n_time = series.shape
    field = torch.full((n_sample, n_time, len(LAT), len(LON)), float("nan"))
    field[:, :, rows, :] = torch.tensor(series, dtype=torch.float32)[:, :, None, None]
    return field.to(get_device())


def _batch(time, prediction, target=None, i_time_start=0) -> InferenceBatchData:
    target = prediction if target is None else target
    return InferenceBatchData(
        prediction={VARIABLE: prediction},
        target={VARIABLE: target},
        time=time,
        i_time_start=i_time_start,
    )


def test_water_year_slices_keep_only_complete_years():
    time = _daily_time((2000, 1, 1), 365 * 3)
    slices = water_year_slices(list(time.values[0]), start_month=10)
    assert len(slices) == 2
    dates = list(time.values[0])
    for sl in slices:
        assert (dates[sl.start].month, dates[sl.start].day) == (10, 1)
        assert (dates[sl.stop].month, dates[sl.stop].day) == (10, 1)


def test_water_year_slices_with_leap_year():
    time = _daily_time((2003, 6, 1), 365 * 3, calendar="standard")
    dates = list(time.values[0])
    slices = water_year_slices(dates, start_month=10)
    lengths = [sl.stop - sl.start for sl in slices]
    assert lengths == [366, 365]


def test_phenology_of_triangular_season():
    days = np.arange(365, dtype=float)
    months = ((np.arange(365) // 30.4).astype(int) % 12) + 1
    values = np.zeros(365)
    values[30:151] = 100 * (np.arange(30, 151) - 30) / 120
    values[151:241] = 100 * (240 - np.arange(151, 241)) / 90
    stats = phenology(values, days, months)
    assert stats["peak"] == pytest.approx(100.0)
    assert stats["meltout_day"] == pytest.approx(231.0)
    assert stats["summer_floor"] == pytest.approx(0.0)
    assert stats["midwinter_mean"] > 0


def test_phenology_of_flat_zero_is_nan():
    stats = phenology(np.zeros(365), np.arange(365.0), np.ones(365, dtype=int))
    assert all(np.isnan(stats[q]) for q in QUANTITIES)


@pytest.mark.parametrize("calendar", ["noleap", "standard"])
def test_recovers_known_season(calendar: str):
    time = _daily_time((2000, 1, 1), 365 * 3 + 10, n_sample=2, calendar=calendar)
    series = triangular_season(time, start_month=10, peak=80.0)
    agg = _aggregator()
    agg.record_batch(_batch(time, _fields(series, rows=slice(2, 4))))
    ds = agg.get_dataset()
    assert ds.sizes["water_year"] == 2
    assert ds.sizes["sample"] == 2
    peak = ds["peak"].sel(source="prediction", region="north").values
    np.testing.assert_allclose(peak, 80.0, rtol=1e-5)
    meltout = ds["meltout_day"].sel(source="prediction", region="north").values
    np.testing.assert_allclose(meltout, 231.0)
    assert (ds["start_year"].sel(region="north").values == [[2000, 2001]] * 2).all()
    logs = agg.get_logs("snow_season")
    assert logs["snow_season/prediction/north/peak"] == pytest.approx(80.0, rel=1e-5)
    assert logs["snow_season/gap/north/peak"] == pytest.approx(0.0, abs=1e-6)
    assert "snow_season/traces" in logs
    assert not any(key.endswith("/meltout_day") for key in logs)


def test_streaming_matches_single_batch_and_drops_initial_condition():
    time = _daily_time((2000, 1, 1), 365 * 3, n_sample=2)
    target = triangular_season(time, 10, 60.0)
    prediction = 1.2 * target
    single = _aggregator()
    single.record_batch(
        _batch(time, _fields(prediction), _fields(target), i_time_start=0)
    )
    streamed = _aggregator()
    for start in range(0, time.sizes["time"], 25):
        stop = min(start + 25, time.sizes["time"])
        streamed.record_batch(
            _batch(
                time.isel(time=slice(start, stop)),
                _fields(prediction[:, start:stop]),
                _fields(target[:, start:stop]),
                i_time_start=start,
            )
        )
    xr.testing.assert_allclose(single.get_dataset(), streamed.get_dataset())
    without_ic = _aggregator()
    without_ic.record_batch(
        _batch(
            time.isel(time=slice(1, None)),
            _fields(prediction[:, 1:]),
            _fields(target[:, 1:]),
            i_time_start=1,
        )
    )
    xr.testing.assert_allclose(single.get_dataset(), without_ic.get_dataset())
    logs = single.get_logs("s")
    assert logs["s/gap/north/peak"] == pytest.approx(0.2 * 60.0, rel=1e-5)


def test_nan_outside_mask_counts_as_zero():
    time = _daily_time((2000, 1, 1), 365 * 3)
    series = triangular_season(time, 10, 50.0)
    masked = _aggregator()
    masked.record_batch(_batch(time, _fields(series, rows=slice(3, 4))))
    zeros = _aggregator()
    field = _fields(series, rows=slice(3, 4))
    field = torch.nan_to_num(field, nan=0.0)
    zeros.record_batch(_batch(time, field))
    xr.testing.assert_allclose(masked.get_dataset(), zeros.get_dataset())
    peak = masked.get_dataset()["peak"].sel(source="prediction", region="north").values
    assert (peak < 50.0).all() and (peak > 0).all()


def test_southern_box_uses_april_start():
    time = _daily_time((2000, 1, 1), 365 * 3)
    series = triangular_season(time, start_month=4, peak=30.0)
    agg = _aggregator(regions=[SOUTH])
    agg.record_batch(_batch(time, _fields(series, rows=slice(0, 2))))
    ds = agg.get_dataset()
    assert ds.sizes["water_year"] == 2
    np.testing.assert_allclose(
        ds["peak"].sel(source="prediction", region="south").values, 30.0, rtol=1e-5
    )


def test_samples_starting_in_different_years():
    long = _daily_time((2000, 1, 1), 365 * 4)
    short = _daily_time((2001, 1, 1), 365 * 4)
    time = xr.concat([long, short], dim="sample")
    series = triangular_season(time, 10, 40.0)
    agg = _aggregator()
    agg.record_batch(_batch(time, _fields(series)))
    ds = agg.get_dataset()
    assert ds.sizes["water_year"] == 3
    assert (
        ds["start_year"].sel(region="north").values
        == [[2000, 2001, 2002], [2001, 2002, 2003]]
    ).all()


def test_variable_absent_records_nothing():
    time = _daily_time((2000, 1, 1), 40)
    agg = _aggregator(variable="missing")
    agg.record_batch(_batch(time, _fields(np.ones(time.shape))))
    assert agg.get_logs("x") == {}
    assert len(agg.get_dataset()) == 0


def test_gather_path_with_mock_distributed():
    time = _daily_time((2000, 1, 1), 365 * 3)
    series = triangular_season(time, 10, 20.0)
    with mock_distributed(world_size=2):
        agg = _aggregator()
        agg.record_batch(_batch(time, _fields(series)))
        ds = agg.get_dataset()
    assert ds.sizes["sample"] == 2


def test_config_validation():
    with pytest.raises(ValueError, match="1-12"):
        SnowSeasonMetricConfig(start_month_northern=13)
    with pytest.raises(ValueError, match="unique"):
        SnowSeasonMetricConfig(regions=[NORTH, NORTH])
    with pytest.raises(ValueError, match="at least one region"):
        SnowSeasonMetricConfig(enabled=True)
    assert SnowSeasonMetricConfig().enabled is False


def _build_context(coords, n_forward_steps: int) -> MetricBuildContext:
    ds_info = DatasetInfo(horizontal_coordinates=coords, timestep=TIMESTEP)
    return MetricBuildContext(
        ops=ds_info.gridded_operations,
        horizontal_coordinates=coords,
        n_timesteps=n_forward_steps + 1,
        n_ic_steps=1,
        timestep=TIMESTEP,
        variable_metadata=None,
        channel_mean_names=None,
        monthly_reference_data=None,
        time_mean_reference_data=None,
        initial_time=_daily_time((2000, 1, 1), 1).isel(time=slice(0, 0)),
    )


def test_build_support_conditions():
    config = SnowSeasonMetricConfig(enabled=True, regions=[NORTH])
    healpix = HEALPixCoordinates(
        face=torch.arange(12), height=torch.arange(4), width=torch.arange(4)
    )
    with pytest.raises(MetricNotSupportedError, match="LatLonCoordinates"):
        config.build(_build_context(healpix, n_forward_steps=1000))
    latlon = LatLonCoordinates(lat=LAT, lon=LON)
    with pytest.raises(MetricNotSupportedError, match="two years"):
        config.build(_build_context(latlon, n_forward_steps=365))
    assert config.build(_build_context(latlon, n_forward_steps=1000)).aggregator


def test_wired_into_evaluator_config():
    names = [m.get_name() for m in InferenceEvaluatorAggregatorConfig()._get_metrics()]
    assert "snow_season" not in names
    config = InferenceEvaluatorAggregatorConfig(
        snow_season=SnowSeasonMetricConfig(enabled=True, regions=[NORTH])
    )
    assert "snow_season" in [m.get_name() for m in config._get_metrics()]
