"""Tests for compute_nino_lead_labels core logic on synthetic data.

Run from this directory: ``python -m pytest test_compute_nino_lead_labels.py``
"""

import numpy as np
import xarray as xr
from compute_nino_lead_labels import (
    advance_year_month,
    build_lead_values,
    compute_nino_lead_labels,
    fme_monthly_index_lookup,
    nino_box_weighted_mean,
    subtract_linear_trend,
)

NINO_LAT = (-5.0, 5.0)
NINO_LON = (190.0, 240.0)


def test_advance_year_month():
    assert advance_year_month(2000, 1, 1) == (2000, 2)
    assert advance_year_month(2000, 12, 1) == (2001, 1)
    assert advance_year_month(2000, 1, 12) == (2001, 1)
    assert advance_year_month(2000, 6, 12) == (2001, 6)
    assert advance_year_month(2000, 1, -1) == (1999, 12)


def test_build_lead_values_mapping_and_nan_tail():
    # index keyed by ym = year*12 + (month-1)
    index = {2000 * 12 + m: float(m) for m in range(12)}  # Jan..Dec 2000
    host_years = np.array([2000, 2000])
    host_months = np.array([1, 12])  # Jan 2000, Dec 2000
    out = build_lead_values(host_years, host_months, index, n_leads=3, first_lead=1)
    # Jan 2000 -> leads Feb, Mar, Apr 2000 -> 1, 2, 3
    np.testing.assert_array_equal(out[0], [1.0, 2.0, 3.0])
    # Dec 2000 -> Jan/Feb/Mar 2001, none in dict -> all NaN
    assert np.all(np.isnan(out[1]))


def _synthetic_host(calendar: str = "noleap") -> xr.Dataset:
    """Monthly host with box SST depending only on year.

    SST inside the box = 273 + (year - 2000); outside the box = 1000 (should be
    excluded by box selection). With 3 years the per-year anomalies are
    -1, 0, +1.
    """
    time = xr.date_range(
        "2000-01-01", periods=36, freq="MS", calendar=calendar, use_cftime=True
    )
    lat = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    lon = np.array([180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0])
    years = np.array([t.year for t in time])
    in_box_lat = (lat >= NINO_LAT[0]) & (lat <= NINO_LAT[1])
    in_box_lon = (lon >= NINO_LON[0]) & (lon <= NINO_LON[1])
    box = in_box_lat[None, :, None] & in_box_lon[None, None, :]
    base = (273.0 + (years - 2000))[:, None, None]
    sst = np.where(box, base, 1000.0).astype(np.float32)
    sst = np.broadcast_to(sst, (len(time), len(lat), len(lon))).copy()
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], sst)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _box_mean(ds: xr.Dataset) -> xr.DataArray:
    return nino_box_weighted_mean(ds["sst"], "lat", "lon", NINO_LAT, NINO_LON).compute()


def test_box_mean_excludes_outside_box():
    ds = _synthetic_host()
    box_mean = _box_mean(ds)
    years = np.array([t.year for t in ds["time"].values])
    np.testing.assert_allclose(box_mean.values, 273.0 + (years - 2000), atol=1e-4)


def test_index_no_running_is_monthly_anomaly():
    # n_running_months=1 -> raw monthly anomaly; here anomaly depends only on
    # year: -1 (2000), 0 (2001), +1 (2002).
    ds = _synthetic_host()
    index = fme_monthly_index_lookup(_box_mean(ds), "time", n_running_months=1)
    np.testing.assert_allclose(index[2000 * 12 + 0], -1.0, atol=1e-5)  # Jan 2000
    np.testing.assert_allclose(index[2001 * 12 + 5], 0.0, atol=1e-5)  # Jun 2001
    np.testing.assert_allclose(index[2002 * 12 + 11], 1.0, atol=1e-5)  # Dec 2002


def test_running_mean_matches_trailing_mean_of_anomalies():
    ds = _synthetic_host()
    box_mean = _box_mean(ds)
    anom = fme_monthly_index_lookup(box_mean, "time", n_running_months=1)
    run5 = fme_monthly_index_lookup(box_mean, "time", n_running_months=5)
    sorted_ym = sorted(anom)
    # First 4 months have no trailing-5 index.
    assert min(run5) == sorted_ym[4]
    assert len(run5) == len(sorted_ym) - 4
    # Each running value equals the trailing 5-month mean of the anomalies.
    for i in range(4, len(sorted_ym)):
        window = [anom[k] for k in sorted_ym[i - 4 : i + 1]]
        np.testing.assert_allclose(
            run5[sorted_ym[i]], float(np.mean(window)), atol=1e-6
        )


def _uniform_warming_host(calendar: str = "noleap") -> xr.Dataset:
    """Host whose SST is spatially uniform with a per-year warming trend.

    Because the Nino box mean equals the tropical mean, the tropical-relative
    index must be ~0 everywhere (pure trend, no ENSO), while the plain index
    carries the -1/0/+1 per-year trend.
    """
    time = xr.date_range(
        "2000-01-01", periods=36, freq="MS", calendar=calendar, use_cftime=True
    )
    lat = np.array([-5.0, 0.0, 5.0])
    lon = np.array([100.0, 200.0, 220.0, 300.0])
    years = np.array([t.year for t in time])
    sst = (273.0 + (years - 2000)).astype(np.float32)[:, None, None]
    sst = np.broadcast_to(sst, (len(time), len(lat), len(lon))).copy()
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], sst)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _compute_uniform(
    ds: xr.Dataset,
    relative_to_tropical: bool,
    linear_detrend: bool = False,
) -> xr.Dataset:
    return compute_nino_lead_labels(
        ds,
        sst_var="sst",
        lat_dim="lat",
        lon_dim="lon",
        time_dim="time",
        n_leads=3,
        first_lead=1,
        lat_bounds=NINO_LAT,
        lon_bounds=NINO_LON,
        clim_start_year_month=None,
        clim_stop_year_month=None,
        n_running_months=1,
        relative_to_tropical=relative_to_tropical,
        linear_detrend=linear_detrend,
    )


def test_subtract_linear_trend_removes_trend_preserves_oscillation():
    n = 240
    t = np.arange(n)
    oscillation = 2.0 * np.sin(2 * np.pi * t / 41.0)  # period incommensurate w/ 12
    series = xr.DataArray(0.02 * t + 5.0 + oscillation, dims=["time"])
    out = subtract_linear_trend(series, "time").values
    # Linear trend removed: residual slope ~ 0.
    assert abs(np.polyfit(t, out, 1)[0]) < 1e-3
    # Oscillation amplitude preserved (~2), unlike a tropical-mean subtraction.
    assert out.max() > 1.5 and out.min() < -1.5
    # A purely linear series detrends to ~0.
    linear = xr.DataArray(3.0 * t - 7.0, dims=["time"])
    np.testing.assert_allclose(
        subtract_linear_trend(linear, "time").values, 0.0, atol=1e-6
    )


def _linear_warming_host(calendar: str = "noleap") -> xr.Dataset:
    """Host whose box SST increases linearly with the time index (uniform in
    space), so a linear detrend removes it exactly."""
    time = xr.date_range(
        "2000-01-01", periods=48, freq="MS", calendar=calendar, use_cftime=True
    )
    lat = np.array([-5.0, 0.0, 5.0])
    lon = np.array([190.0, 210.0, 230.0])
    t = np.arange(len(time))
    sst = (273.0 + 0.05 * t).astype(np.float32)[:, None, None]
    sst = np.broadcast_to(sst, (len(time), len(lat), len(lon))).copy()
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], sst)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_linear_detrend_removes_uniform_trend():
    ds = _linear_warming_host()
    plain = _compute_uniform(ds, relative_to_tropical=False)
    detrended = _compute_uniform(ds, relative_to_tropical=False, linear_detrend=True)
    plain_vals = plain["nino34_lead_01"].isel(lat=0, lon=0).values
    det_vals = detrended["nino34_lead_01"].isel(lat=0, lon=0).values
    assert np.nanmax(np.abs(plain_vals)) > 0.5
    np.testing.assert_allclose(
        np.nan_to_num(det_vals), np.zeros_like(det_vals), atol=1e-4
    )


def test_relative_to_tropical_removes_uniform_trend():
    ds = _uniform_warming_host()
    plain = _compute_uniform(ds, relative_to_tropical=False)
    relative = _compute_uniform(ds, relative_to_tropical=True)
    # Plain index carries the trend (nonzero anomalies), relative index removes it.
    plain_vals = plain["nino34_lead_01"].isel(lat=0, lon=0).values
    rel_vals = relative["nino34_lead_01"].isel(lat=0, lon=0).values
    assert np.nanmax(np.abs(plain_vals)) > 0.5
    np.testing.assert_allclose(
        np.nan_to_num(rel_vals), np.zeros_like(rel_vals), atol=1e-5
    )


def test_end_to_end_no_running_values_and_coords():
    ds = _synthetic_host()
    out = compute_nino_lead_labels(
        ds,
        sst_var="sst",
        lat_dim="lat",
        lon_dim="lon",
        time_dim="time",
        n_leads=12,
        first_lead=1,
        lat_bounds=NINO_LAT,
        lon_bounds=NINO_LON,
        clim_start_year_month=None,
        clim_stop_year_month=None,
        n_running_months=1,
    )
    assert [f"nino34_lead_{k:02d}" for k in range(1, 13)] == sorted(out.data_vars)
    assert list(out["time"].values) == list(ds["time"].values)
    np.testing.assert_array_equal(out["lat"].values, ds["lat"].values)
    np.testing.assert_array_equal(out["lon"].values, ds["lon"].values)

    lead1 = out["nino34_lead_01"]
    assert float(lead1.isel(time=0).std()) == 0.0  # constant across space
    # Jan 2000 -> Feb 2000 (year 2000) -> anomaly -1
    np.testing.assert_allclose(float(lead1.isel(time=0, lat=0, lon=0)), -1.0, atol=1e-4)
    # lead 12 from Jan 2000 -> Jan 2001 (year 2001) -> anomaly 0
    np.testing.assert_allclose(
        float(out["nino34_lead_12"].isel(time=0, lat=0, lon=0)), 0.0, atol=1e-4
    )
    # tail: last host time (Dec 2002) has no +1 month target -> NaN
    assert np.isnan(float(lead1.isel(time=-1, lat=0, lon=0)))


def test_end_to_end_default_running_mean_nans_early_targets():
    ds = _synthetic_host()
    out = compute_nino_lead_labels(
        ds,
        sst_var="sst",
        lat_dim="lat",
        lon_dim="lon",
        time_dim="time",
        n_leads=12,
        first_lead=1,
        lat_bounds=NINO_LAT,
        lon_bounds=NINO_LON,
        clim_start_year_month=None,
        clim_stop_year_month=None,
        # default FME running-mean window
    )
    lead1 = out["nino34_lead_01"]
    # Jan 2000 -> Feb 2000 is month index 1 (< 5-month window) -> NaN index
    assert np.isnan(float(lead1.isel(time=0, lat=0, lon=0)))
    # A later host time has a valid running-mean target, constant across space
    valid = lead1.isel(time=12)
    assert np.isfinite(float(valid.isel(lat=0, lon=0)))
    np.testing.assert_allclose(float(valid.std()), 0.0, atol=1e-5)
