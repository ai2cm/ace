"""Tests for processing.py — shared CMIP6 processing utilities.

Requires xarray, numpy, and xesmf (for regridding tests).
"""

import numpy as np
import pytest
import xarray as xr
from config import RegridConfig, ResolvedDatasetConfig, TimeWindow
from processing import (
    BOUNDS_NAMES,
    UNSTRUCTURED_METHOD,
    DuplicateTimestampsError,
    SimulationBoundaryError,
    _has_bounds,
    apply_output_renames,
    apply_target_land_mask,
    apply_time_subset,
    causal_monthly_to_daily,
    clamp_static_fractions,
    clip_date_for_calendar,
    compute_below_surface_mask,
    compute_derived_layer_T,
    compute_ocean_fraction,
    emit_mask_and_fill,
    fill_derived_layer_T,
    fill_horizontal_diffuse,
    finalize_surface_and_ocean_variable,
    flatten_plev_variables,
    is_unstructured_source,
    make_regridder,
    nearest_above_fill,
    normalize_plev,
    normalize_regrid_source,
    regrid_variables,
    resolve_time_duplicates,
    run_sanity_checks,
    validate_cell_methods,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(**overrides) -> ResolvedDatasetConfig:
    from config import FillConfig

    defaults = dict(
        source_id="TEST",
        experiment="historical",
        variant_label="r1i1p1f1",
        core_variables=["ua", "va", "hus", "zg"],
        optional_variables=["tas"],
        surface_and_ocean_variables=["amon_ts"],
        static_variables=["orog"],
        max_core_missing=0,
        time_subset={"historical": TimeWindow("2010-01-01", "2010-12-31")},
        target_grid=type("TG", (), {"name": "F22.5"})(),
        regrid=RegridConfig(),
        fill=FillConfig(),
        chunking=type("CC", (), {"time": 365, "use_sharding": False})(),
    )
    defaults.update(overrides)
    return ResolvedDatasetConfig(**defaults)


def _rectilinear_ds(
    nlat: int = 10,
    nlon: int = 20,
    ntime: int = 5,
    variables: list[str] | None = None,
    with_bounds: bool = True,
    with_plev: bool = False,
    plev_values: list | None = None,
) -> xr.Dataset:
    """Create a synthetic rectilinear dataset mimicking CMIP6 structure."""
    if variables is None:
        variables = ["tas"]

    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(2, 358, nlon)
    time = xr.date_range("2010-01-01", periods=ntime, freq="D", calendar="noleap")

    coords: dict = {"lat": lat, "lon": lon, "time": time}
    dims = ["time", "lat", "lon"]

    if with_plev:
        if plev_values is None:
            plev_values = [100000, 85000, 70000, 50000, 25000, 10000, 5000, 1000]
        coords["plev"] = np.array(plev_values, dtype=np.float64)
        dims = ["time", "plev", "lat", "lon"]

    data_vars = {}
    shape = tuple(len(coords[d]) for d in dims)
    rng = np.random.default_rng(42)
    for v in variables:
        data_vars[v] = xr.DataArray(
            rng.standard_normal(shape).astype(np.float32),
            dims=dims,
            attrs={"cell_methods": "time: mean"},
        )

    ds = xr.Dataset(data_vars, coords=coords)

    if with_bounds:
        dlat = np.diff(lat).mean()
        lat_bnds = np.column_stack([lat - dlat / 2, lat + dlat / 2])
        lat_bnds = np.clip(lat_bnds, -90, 90)
        dlon = np.diff(lon).mean()
        lon_bnds = np.column_stack([lon - dlon / 2, lon + dlon / 2])
        ds["lat_bnds"] = xr.DataArray(lat_bnds, dims=("lat", "bnds"))
        ds["lon_bnds"] = xr.DataArray(lon_bnds, dims=("lon", "bnds"))

    return ds


# ---------------------------------------------------------------------------
# normalize_plev
# ---------------------------------------------------------------------------


def test_normalize_plev_reverses_ascending():
    ds = _rectilinear_ds(with_plev=True, plev_values=[1000, 5000, 10000])
    result = normalize_plev(ds)
    assert list(result["plev"].values) == [10000, 5000, 1000]


def test_normalize_plev_noop_if_descending():
    plev = [100000, 50000, 10000]
    ds = _rectilinear_ds(with_plev=True, plev_values=plev)
    result = normalize_plev(ds)
    assert list(result["plev"].values) == plev


def test_normalize_plev_noop_without_plev():
    ds = _rectilinear_ds(with_plev=False)
    result = normalize_plev(ds)
    assert "plev" not in result.dims


# ---------------------------------------------------------------------------
# resolve_time_duplicates
# ---------------------------------------------------------------------------


def test_resolve_no_duplicates():
    ds = _rectilinear_ds(ntime=5)
    result, msg = resolve_time_duplicates(ds, "tas", allow_dedupe=True)
    assert msg == ""
    assert result.sizes["time"] == 5


def test_resolve_identical_duplicates():
    ds = _rectilinear_ds(ntime=3)
    dup = xr.concat([ds, ds.isel(time=[0])], dim="time", data_vars="minimal")
    result, msg = resolve_time_duplicates(dup, "tas", allow_dedupe=True)
    assert "deduplicated" in msg
    assert result.sizes["time"] == 3


def test_resolve_raises_without_allow_dedupe():
    ds = _rectilinear_ds(ntime=3)
    dup = xr.concat([ds, ds.isel(time=[0])], dim="time", data_vars="minimal")
    with pytest.raises(DuplicateTimestampsError):
        resolve_time_duplicates(dup, "tas", allow_dedupe=False)


def test_resolve_raises_on_boundary_mismatch():
    ds = _rectilinear_ds(ntime=3)
    ds2 = ds.copy(deep=True)
    ds2["tas"].values[0] += 999.0
    dup = xr.concat([ds, ds2.isel(time=[0])], dim="time", data_vars="minimal")
    with pytest.raises(SimulationBoundaryError):
        resolve_time_duplicates(dup, "tas", allow_dedupe=True)


# ---------------------------------------------------------------------------
# validate_cell_methods
# ---------------------------------------------------------------------------


def test_validate_cell_methods_all_pass():
    ds = _rectilinear_ds(variables=["ua", "va"])
    assert validate_cell_methods(ds, ["ua", "va"]) == []


def test_validate_cell_methods_missing_attr():
    ds = _rectilinear_ds(variables=["ua"])
    del ds["ua"].attrs["cell_methods"]
    assert validate_cell_methods(ds, ["ua"]) == ["ua"]


def test_validate_cell_methods_ignores_absent_variables():
    ds = _rectilinear_ds(variables=["ua"])
    assert validate_cell_methods(ds, ["ua", "va"]) == []


# ---------------------------------------------------------------------------
# normalize_regrid_source + _has_bounds
# ---------------------------------------------------------------------------


def test_has_bounds_with_lon_bnds():
    ds = _rectilinear_ds(with_bounds=True)
    assert not _has_bounds(ds)
    norm = normalize_regrid_source(ds)
    assert _has_bounds(norm)


def test_has_bounds_without_bounds():
    ds = _rectilinear_ds(with_bounds=False)
    assert not _has_bounds(ds)
    norm = normalize_regrid_source(ds)
    assert not _has_bounds(norm)


def test_normalize_converts_2d_bounds_to_vertices():
    ds = _rectilinear_ds(nlat=10, nlon=20, with_bounds=True)
    norm = normalize_regrid_source(ds)
    assert norm["lat_b"].shape == (11,)
    assert norm["lon_b"].shape == (21,)


# ---------------------------------------------------------------------------
# make_regridder — bilinear fallback
# ---------------------------------------------------------------------------


def test_make_regridder_conservative_with_bounds():
    from grid import make_target_grid

    ds = _rectilinear_ds(nlat=10, nlon=20, with_bounds=True, ntime=1)
    target = make_target_grid("F22.5")
    regridder, method = make_regridder(ds, target, "conservative")
    assert method == "conservative"


def test_make_regridder_falls_back_to_bilinear_without_bounds():
    from grid import make_target_grid

    ds = _rectilinear_ds(nlat=10, nlon=20, with_bounds=False, ntime=1)
    target = make_target_grid("F22.5")
    regridder, method = make_regridder(ds, target, "conservative")
    assert method == "bilinear"


def test_make_regridder_bilinear_stays_bilinear():
    from grid import make_target_grid

    ds = _rectilinear_ds(nlat=10, nlon=20, with_bounds=False, ntime=1)
    target = make_target_grid("F22.5")
    regridder, method = make_regridder(ds, target, "bilinear")
    assert method == "bilinear"


# ---------------------------------------------------------------------------
# regrid_variables — bounds carried through subsetting
# ---------------------------------------------------------------------------


def test_regrid_preserves_bounds_for_conservative():
    """Bounds variables (lon_bnds, lat_bnds) must survive the ds[vars_]
    subsetting inside regrid_variables so conservative regridding works.

    Uses make_regridder directly to verify method selection, since xesmf's
    conservative regridder is sensitive to grid geometry details that are
    hard to replicate with small synthetic grids.
    """
    from grid import make_target_grid

    ds = _rectilinear_ds(nlat=10, nlon=20, ntime=3, variables=["pr"], with_bounds=True)
    target = make_target_grid("F22.5")

    bounds_to_keep = [v for v in ds.data_vars if v in BOUNDS_NAMES]
    sub = ds[["pr"] + bounds_to_keep]

    regridder, actual_method = make_regridder(sub, target, "conservative")
    assert actual_method == "conservative"

    # Without bounds, it should fall back
    sub_no_bounds = ds[["pr"]]
    _, fallback_method = make_regridder(sub_no_bounds, target, "conservative")
    assert fallback_method == "bilinear"


def test_regrid_bilinear_without_bounds():
    from grid import make_target_grid

    ds = _rectilinear_ds(nlat=10, nlon=20, ntime=3, variables=["ua"], with_bounds=False)
    cfg = _make_cfg()
    target = make_target_grid("F22.5")
    result, methods = regrid_variables(ds, target, cfg)
    assert methods["ua"] == "bilinear"


def test_regrid_mixed_methods():
    """With bounds present, flux vars get conservative, state vars get bilinear."""
    from grid import make_target_grid

    ds = _rectilinear_ds(
        nlat=10, nlon=20, ntime=3, variables=["ua", "pr"], with_bounds=True
    )
    cfg = _make_cfg()
    target = make_target_grid("F22.5")
    result, methods = regrid_variables(ds, target, cfg)
    assert methods["ua"] == "bilinear"
    assert methods["pr"] == "conservative"
    assert result["ua"].shape == result["pr"].shape


def test_regrid_flux_only_with_bounds():
    """Regridding a single flux variable with bounds must not crash.

    Bounds get classified as bilinear data vars if included in
    by_method, causing an empty-dataset regrid that fails with
    'cannot rename lat_new'.
    """
    from grid import make_target_grid

    ds = _rectilinear_ds(nlat=10, nlon=20, ntime=3, variables=["pr"], with_bounds=True)
    cfg = _make_cfg()
    target = make_target_grid("F22.5")
    result, methods = regrid_variables(ds, target, cfg)
    assert methods["pr"] == "conservative"
    assert "lat_bnds" not in result.data_vars
    assert "lon_bnds" not in result.data_vars


# ---------------------------------------------------------------------------
# flatten_plev_variables
# ---------------------------------------------------------------------------


def test_flatten_plev_creates_named_variables():
    ds = _rectilinear_ds(
        ntime=3,
        with_plev=True,
        variables=["ua"],
        plev_values=[100000, 50000, 10000],
    )
    result = flatten_plev_variables(ds)
    assert "ua1000" in result.data_vars
    assert "ua500" in result.data_vars
    assert "ua100" in result.data_vars
    assert "ua" not in result.data_vars
    assert "plev" not in result.dims


def test_flatten_plev_preserves_2d_variables():
    ds = _rectilinear_ds(ntime=3, variables=["tas"])
    result = flatten_plev_variables(ds)
    assert "tas" in result.data_vars


# ---------------------------------------------------------------------------
# run_sanity_checks
# ---------------------------------------------------------------------------


def test_sanity_checks_pass_for_reasonable_data():
    ds = _rectilinear_ds(ntime=3, variables=["tas"])
    ds["tas"].values[:] = 280.0
    warnings = run_sanity_checks(ds)
    assert not any("tas" in w for w in warnings)


def test_sanity_checks_flag_out_of_range():
    ds = _rectilinear_ds(ntime=3, variables=["tas"])
    ds["tas"].values[:] = -1000.0
    warnings = run_sanity_checks(ds)
    assert any("tas" in w for w in warnings)


# ---------------------------------------------------------------------------
# clamp_static_fractions
# ---------------------------------------------------------------------------


def test_clamp_static_fractions_clips_overshoot_and_renames():
    ds = xr.Dataset(
        {"sftlf": (("lat", "lon"), np.array([[0.0, 50.0], [100.0, 114.0]]))},
    )
    clamped, warnings = clamp_static_fractions(ds)
    # sftlf is renamed to land_fraction and rescaled to [0, 1].
    assert "sftlf" not in clamped.data_vars
    assert "land_fraction" in clamped.data_vars
    assert float(clamped["land_fraction"].max()) == 1.0
    assert float(clamped["land_fraction"].min()) == 0.0
    assert clamped["land_fraction"].attrs["original_name"] == "sftlf"
    assert any("sftlf" in w and "114" in w for w in warnings)


def test_clamp_static_fractions_silent_when_in_range():
    ds = xr.Dataset(
        {"sftlf": (("lat", "lon"), np.array([[0.0, 50.0], [99.9, 100.0]]))},
    )
    clamped, warnings = clamp_static_fractions(ds)
    assert warnings == []
    np.testing.assert_allclose(
        clamped["land_fraction"].values, ds["sftlf"].values / 100.0
    )


def test_clamp_static_fractions_noop_without_sftlf():
    ds = xr.Dataset({"orog": (("lat", "lon"), np.zeros((2, 2)))})
    clamped, warnings = clamp_static_fractions(ds)
    assert warnings == []
    assert "sftlf" not in clamped.data_vars
    assert "land_fraction" not in clamped.data_vars


# ---------------------------------------------------------------------------
# compute_below_surface_mask
# ---------------------------------------------------------------------------


def test_below_surface_mask_from_nan_union():
    ds = _rectilinear_ds(ntime=3, with_plev=True, variables=["ua", "va", "hus", "zg"])
    ds["ua"].values[:, -1, :, :] = np.nan
    mask, source = compute_below_surface_mask(ds, orog=None)
    assert source == "nan_union"
    assert mask is not None
    assert mask.dtype == np.uint8
    assert int(mask.isel(plev=-1).sum()) > 0


def test_below_surface_mask_no_nans_no_orog():
    ds = _rectilinear_ds(ntime=3, with_plev=True, variables=["ua", "va", "hus", "zg"])
    mask, source = compute_below_surface_mask(ds, orog=None)
    assert source == "none"
    assert mask is None


def test_below_surface_mask_fallback_to_orog():
    ds = _rectilinear_ds(ntime=3, with_plev=True, variables=["ua", "va", "hus", "zg"])
    orog = xr.DataArray(
        np.full((10, 20), 5000.0, dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
    )
    ds["zg"].values[:] = np.linspace(0, 30000, 8)[None, :, None, None]
    mask, source = compute_below_surface_mask(ds, orog)
    assert source == "orog_static"
    assert mask is not None


# ---------------------------------------------------------------------------
# nearest_above_fill
# ---------------------------------------------------------------------------


def test_nearest_above_fill_fills_bottom_level():
    ds = _rectilinear_ds(
        ntime=2,
        nlat=3,
        nlon=4,
        with_plev=True,
        variables=["ua"],
        plev_values=[100000, 50000, 10000],
    )
    ds["ua"].values[:] = 10.0
    ds["ua"].values[:, 0, :, :] = np.nan

    mask = xr.zeros_like(ds["ua"], dtype=np.uint8)
    mask.values[:, 0, :, :] = 1

    filled = nearest_above_fill(ds["ua"], mask)
    assert not np.isnan(filled.values).any()
    np.testing.assert_allclose(filled.isel(plev=0).values, 10.0)


def test_nearest_above_fill_multiple_levels():
    ds = _rectilinear_ds(
        ntime=1,
        nlat=2,
        nlon=2,
        with_plev=True,
        variables=["zg"],
        plev_values=[100000, 85000, 50000, 10000],
    )
    ds["zg"].values[:] = [[[[1]], [[2]], [[3]], [[4]]]]
    ds["zg"].values[:, :2, :, :] = np.nan

    mask = xr.zeros_like(ds["zg"], dtype=np.uint8)
    mask.values[:, :2, :, :] = 1

    filled = nearest_above_fill(ds["zg"], mask)
    assert not np.isnan(filled.values).any()
    np.testing.assert_allclose(filled.isel(plev=0).values, 3.0)
    np.testing.assert_allclose(filled.isel(plev=1).values, 3.0)


# ---------------------------------------------------------------------------
# compute_derived_layer_T
# ---------------------------------------------------------------------------


def test_derived_layer_T_shape_and_names():
    plev_pa = [100000, 85000, 70000, 50000, 25000, 10000, 5000, 1000]
    ds = _rectilinear_ds(
        ntime=3,
        nlat=4,
        nlon=8,
        with_plev=True,
        variables=["zg", "hus"],
        plev_values=plev_pa,
    )
    ds["zg"].values[:] = np.linspace(0, 30000, 8)[None, :, None, None]
    ds["hus"].values[:] = 0.005
    result = compute_derived_layer_T(ds)
    assert "ta_derived_layer" in result.data_vars
    assert result["ta_derived_layer"].sizes["plev_layer"] == 7
    assert result["ta_derived_layer"].sizes["time"] == 3


def test_derived_layer_T_reasonable_values():
    plev_pa = [100000, 50000, 10000]
    ds = _rectilinear_ds(
        ntime=1,
        nlat=2,
        nlon=2,
        with_plev=True,
        variables=["zg", "hus"],
        plev_values=plev_pa,
    )
    ds["zg"].values[0, 0, :, :] = 0
    ds["zg"].values[0, 1, :, :] = 5500
    ds["zg"].values[0, 2, :, :] = 16000
    ds["hus"].values[:] = 0.005
    result = compute_derived_layer_T(ds)
    ta = result["ta_derived_layer"]
    assert ta.min() > 150.0
    assert ta.max() < 350.0


# ---------------------------------------------------------------------------
# fill_derived_layer_T
# ---------------------------------------------------------------------------


def test_fill_derived_layer_T_fills_bottom_layers():
    plev_pa = [100000, 50000, 10000]
    ds = _rectilinear_ds(
        ntime=1,
        nlat=2,
        nlon=2,
        with_plev=True,
        variables=["zg", "hus"],
        plev_values=plev_pa,
    )
    ds["zg"].values[0, 0, :, :] = 0
    ds["zg"].values[0, 1, :, :] = 5500
    ds["zg"].values[0, 2, :, :] = 16000
    ds["hus"].values[:] = 0.005
    ds = compute_derived_layer_T(ds)

    mask = xr.zeros_like(ds["zg"], dtype=np.uint8)
    mask.values[:, 0, :, :] = 1

    result = fill_derived_layer_T(ds, mask)
    ta = result["ta_derived_layer"]
    ta_bottom = ta.isel(plev_layer=0).values
    ta_top = ta.isel(plev_layer=1).values
    np.testing.assert_allclose(ta_bottom, ta_top)


# ---------------------------------------------------------------------------
# clip_date_for_calendar
# ---------------------------------------------------------------------------


def test_clip_date_standard_calendar():
    assert clip_date_for_calendar("2010-01-31", "standard") == "2010-01-31"


def test_clip_date_360_day_clips():
    assert clip_date_for_calendar("2010-01-31", "360_day") == "2010-01-30"


def test_clip_date_360_day_noop_for_30():
    assert clip_date_for_calendar("2010-06-30", "360_day") == "2010-06-30"


# ---------------------------------------------------------------------------
# apply_time_subset
# ---------------------------------------------------------------------------


def test_apply_time_subset_selects_window():
    ds = _rectilinear_ds(ntime=365)
    cfg = _make_cfg(
        experiment="historical",
        time_subset={
            "historical": TimeWindow("2010-01-01", "2010-01-31"),
        },
    )
    result = apply_time_subset(ds, cfg)
    assert result.sizes["time"] == 31


def test_apply_time_subset_noop_when_no_window():
    ds = _rectilinear_ds(ntime=10)
    cfg = _make_cfg(experiment="historical", time_subset={})
    result = apply_time_subset(ds, cfg)
    assert result.sizes["time"] == 10


# ---------------------------------------------------------------------------
# fill_horizontal_diffuse
# ---------------------------------------------------------------------------


def test_fill_horizontal_diffuse_fills_continent():
    ny, nx = 45, 90
    field = np.full((ny, nx), 280.0)
    field[10:20, 30:60] = np.nan  # big continent
    da = xr.DataArray(field, dims=("lat", "lon"))
    filled = fill_horizontal_diffuse(da, max_iterations=100)
    assert int(np.isnan(filled.values).sum()) == 0
    # Originally-valid cells are unchanged.
    assert float(filled.values[0, 0]) == 280.0


def test_fill_horizontal_diffuse_preserves_originally_valid():
    ny, nx = 20, 40
    field = np.linspace(0, 1, ny * nx).reshape(ny, nx) * 100
    valid_mask = np.ones_like(field, dtype=bool)
    field2 = field.copy()
    field2[5:15, 10:30] = np.nan
    valid_mask[5:15, 10:30] = False
    da = xr.DataArray(field2, dims=("lat", "lon"))
    filled = fill_horizontal_diffuse(da, max_iterations=200)
    np.testing.assert_array_equal(filled.values[valid_mask], field[valid_mask])


def test_fill_horizontal_diffuse_3d_per_timestep():
    nt, ny, nx = 3, 10, 20
    arr = np.full((nt, ny, nx), 250.0)
    arr[0, :, 5:15] = np.nan
    arr[1, :5, :] = np.nan
    arr[2, 5:, :] = np.nan
    da = xr.DataArray(arr, dims=("time", "lat", "lon"))
    filled = fill_horizontal_diffuse(da, max_iterations=50)
    assert int(np.isnan(filled.values).sum()) == 0


def test_fill_horizontal_diffuse_all_nan_falls_back_to_zero():
    da = xr.DataArray(np.full((10, 10), np.nan), dims=("lat", "lon"))
    filled = fill_horizontal_diffuse(da)
    assert int(np.isnan(filled.values).sum()) == 0
    assert float(filled.max()) == 0.0
    assert float(filled.min()) == 0.0


# ---------------------------------------------------------------------------
# causal_monthly_to_daily
# ---------------------------------------------------------------------------


def _make_monthly(values: list[float], calendar: str = "noleap"):
    import cftime

    if calendar == "noleap":
        times = [cftime.DatetimeNoLeap(2010, m, 15) for m in range(1, len(values) + 1)]
    else:
        times = [
            cftime.DatetimeGregorian(2010, m, 15) for m in range(1, len(values) + 1)
        ]
    return xr.DataArray(np.array(values), dims=("time",), coords={"time": times})


def _make_annual(values: list[float]):
    import cftime

    times = [cftime.DatetimeNoLeap(2008 + i, 7, 1, 12) for i in range(len(values))]
    return xr.DataArray(np.array(values), dims=("time",), coords={"time": times})


def test_causal_annual_to_daily_uses_previous_year():
    from processing import causal_annual_to_daily

    annual = _make_annual([100.0, 200.0, 300.0, 400.0])  # 2008, 2009, 2010, 2011
    daily_times = xr.date_range(
        "2010-06-01", "2010-06-05", freq="D", use_cftime=True, calendar="noleap"
    )
    daily = xr.DataArray(daily_times, dims=("time",))
    out = causal_annual_to_daily(annual, daily)
    # 2010 days take 2009's value = 200.0
    assert float(out.min()) == 200.0
    assert float(out.max()) == 200.0


def test_causal_annual_to_daily_falls_back_at_start():
    # Annual data starts at 2015; daily query starts at 2010 → no prior
    # year available, fall back to first annual value.
    import cftime
    from processing import causal_annual_to_daily

    times = [cftime.DatetimeNoLeap(2015 + i, 7, 1, 12) for i in range(3)]
    annual = xr.DataArray(
        np.array([500.0, 600.0, 700.0]),
        dims=("time",),
        coords={"time": times},
    )
    daily_times = xr.date_range(
        "2010-01-01", "2010-01-05", freq="D", use_cftime=True, calendar="noleap"
    )
    daily = xr.DataArray(daily_times, dims=("time",))
    out = causal_annual_to_daily(annual, daily)
    assert float(out.isel(time=0)) == 500.0  # first-annual fallback


def test_causal_monthly_to_daily_uses_previous_month():
    monthly = _make_monthly([100.0, 200.0, 300.0, 400.0])  # Jan, Feb, Mar, Apr
    daily_times = xr.date_range(
        "2010-03-01", "2010-03-31", freq="D", use_cftime=True, calendar="noleap"
    )
    daily = xr.DataArray(daily_times, dims=("time",))
    out = causal_monthly_to_daily(monthly, daily)
    # All of March → Feb mean = 200
    assert float(out.min()) == 200.0
    assert float(out.max()) == 200.0


def test_causal_monthly_to_daily_falls_back_at_start():
    # Daily window starts in January, but monthly data starts in February
    # → no December available, fall back to the first monthly value.
    import cftime

    times = [cftime.DatetimeNoLeap(2010, m, 15) for m in range(2, 5)]
    monthly = xr.DataArray(
        np.array([200.0, 300.0, 400.0]), dims=("time",), coords={"time": times}
    )
    daily_times = xr.date_range(
        "2010-01-01", "2010-01-31", freq="D", use_cftime=True, calendar="noleap"
    )
    daily = xr.DataArray(daily_times, dims=("time",))
    out = causal_monthly_to_daily(monthly, daily)
    assert float(out.isel(time=0)) == 200.0  # first monthly fallback


def test_causal_monthly_to_daily_spans_year_boundary():
    import cftime

    # Monthly data: Dec 2009, Jan 2010
    times = [
        cftime.DatetimeNoLeap(2009, 12, 15),
        cftime.DatetimeNoLeap(2010, 1, 15),
    ]
    monthly = xr.DataArray(
        np.array([500.0, 600.0]), dims=("time",), coords={"time": times}
    )
    daily_times = xr.date_range(
        "2010-01-01", "2010-01-31", freq="D", use_cftime=True, calendar="noleap"
    )
    daily = xr.DataArray(daily_times, dims=("time",))
    out = causal_monthly_to_daily(monthly, daily)
    # All January days take December's mean.
    assert float(out.min()) == 500.0
    assert float(out.max()) == 500.0


# ---------------------------------------------------------------------------
# emit_mask_and_fill
# ---------------------------------------------------------------------------


def test_emit_mask_and_fill_static_pattern_collapses():
    """Time-invariant NaN pattern → 2D mask."""
    nt, ny, nx = 3, 10, 20
    arr = np.full((nt, ny, nx), 280.0)
    arr[:, 5:, :] = np.nan  # static land mask
    da = xr.DataArray(arr, dims=("time", "lat", "lon"))
    filled, mask = emit_mask_and_fill(da, fill_iterations=20)
    assert mask.dims == ("lat", "lon")
    assert mask.shape == (ny, nx)
    assert int(np.isnan(filled.values).sum()) == 0


def test_emit_mask_and_fill_time_varying_pattern_kept_3d():
    """Time-varying NaN pattern → 3D mask."""
    nt, ny, nx = 3, 10, 20
    arr = np.full((nt, ny, nx), 250.0)
    for t in range(nt):
        arr[t, : (t + 1) * 2, :] = np.nan
    da = xr.DataArray(arr, dims=("time", "lat", "lon"))
    filled, mask = emit_mask_and_fill(da)
    assert mask.dims == ("time", "lat", "lon")
    assert int(np.isnan(filled.values).sum()) == 0


def test_emit_mask_and_fill_no_nan_returns_all_valid_mask():
    da = xr.DataArray(np.ones((3, 5, 10)) * 300.0, dims=("time", "lat", "lon"))
    filled, mask = emit_mask_and_fill(da)
    assert int(mask.sum()) == 5 * 10
    assert float(filled.min()) == 300.0


# ---------------------------------------------------------------------------
# finalize_surface_and_ocean_variable
# ---------------------------------------------------------------------------


def test_finalize_surface_and_ocean_atmos_surface_no_mask():
    """atmos_surface kind should NOT emit a _mask companion."""
    from config import SURFACE_AND_OCEAN_BY_OUTPUT

    h = SURFACE_AND_OCEAN_BY_OUTPUT["eday_ts"]
    daily_times = xr.date_range(
        "2010-01-01", "2010-01-05", freq="D", use_cftime=True, calendar="noleap"
    )
    da = xr.DataArray(
        np.full((5, 4, 8), 280.0),
        dims=("time", "lat", "lon"),
        coords={"time": daily_times},
    )
    outputs = finalize_surface_and_ocean_variable(
        da, h, xr.DataArray(daily_times, dims=("time",))
    )
    assert set(outputs.keys()) == {"eday_ts"}


def test_finalize_surface_and_ocean_ocean_emits_mask():
    """ocean_surface kind emits {name}_mask companion."""
    from config import SURFACE_AND_OCEAN_BY_OUTPUT

    h = SURFACE_AND_OCEAN_BY_OUTPUT["oday_tos"]
    daily_times = xr.date_range(
        "2010-01-01", "2010-01-05", freq="D", use_cftime=True, calendar="noleap"
    )
    arr = np.full((5, 6, 12), 290.0)
    arr[:, 3:, :] = np.nan  # land
    da = xr.DataArray(arr, dims=("time", "lat", "lon"), coords={"time": daily_times})
    outputs = finalize_surface_and_ocean_variable(
        da, h, xr.DataArray(daily_times, dims=("time",))
    )
    assert set(outputs.keys()) == {"oday_tos", "oday_tos_mask"}
    assert int(np.isnan(outputs["oday_tos"].values).sum()) == 0


def test_finalize_surface_and_ocean_monthly_causal_maps_correctly():
    """monthly_causal cadence applies previous-month mapping."""
    import cftime
    from config import SURFACE_AND_OCEAN_BY_OUTPUT

    h = SURFACE_AND_OCEAN_BY_OUTPUT["amon_ts"]
    monthly_times = [cftime.DatetimeNoLeap(2010, m, 15) for m in range(1, 4)]
    monthly = xr.DataArray(
        np.stack([np.full((4, 8), v) for v in [100.0, 200.0, 300.0]]),
        dims=("time", "lat", "lon"),
        coords={"time": monthly_times},
    )
    daily_times = xr.date_range(
        "2010-03-01", "2010-03-05", freq="D", use_cftime=True, calendar="noleap"
    )
    outputs = finalize_surface_and_ocean_variable(
        monthly, h, xr.DataArray(daily_times, dims=("time",))
    )
    # March days get February's mean = 200.
    assert float(outputs["amon_ts"].mean()) == 200.0


# ---------------------------------------------------------------------------
# Unstructured-source detection and regridding
# ---------------------------------------------------------------------------


def _unstructured_ds(ncells: int = 64, ntime: int = 3) -> xr.Dataset:
    """Synthetic FESOM-shaped dataset: 1D ``ncells`` with 1D lat/lon."""
    rng = np.random.default_rng(0)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, ncells)))
    lon = rng.uniform(0, 360, ncells)
    time = xr.date_range("2010-01-01", periods=ntime, freq="D", calendar="noleap")
    data = rng.standard_normal((ntime, ncells)).astype(np.float32)
    return xr.Dataset(
        {"tos": (("time", "ncells"), data)},
        coords={
            "lat": ("ncells", lat),
            "lon": ("ncells", lon),
            "time": time,
        },
    )


def test_is_unstructured_source_detects_ncells():
    ds = _unstructured_ds()
    assert is_unstructured_source(ds)


def test_is_unstructured_source_rejects_rectilinear():
    ds = _rectilinear_ds(nlat=8, nlon=16, ntime=2, with_bounds=False)
    assert not is_unstructured_source(ds)


def test_is_unstructured_source_rejects_no_latlon():
    ds = xr.Dataset({"x": (("a",), np.zeros(3))})
    assert not is_unstructured_source(ds)


def test_is_unstructured_source_rejects_2d_curvilinear():
    """Curvilinear grids (lat/lon are 2D, like ORCA tripolar) are
    NOT what locstream_in handles — they should go through the
    standard structured path with bounds."""
    ds = xr.Dataset(
        {"tos": (("time", "y", "x"), np.zeros((2, 4, 5)))},
        coords={
            "lat": (("y", "x"), np.zeros((4, 5))),
            "lon": (("y", "x"), np.zeros((4, 5))),
        },
    )
    assert not is_unstructured_source(ds)


def test_make_regridder_unstructured_uses_locstream():
    from grid import make_target_grid

    ds = _unstructured_ds(ncells=200, ntime=2)
    target = make_target_grid("F22.5")
    regridder, method = make_regridder(ds, target, "bilinear")
    # The method is reported with the locstream-aware sentinel so the
    # caller can detect and apply the target land mask.
    assert method == UNSTRUCTURED_METHOD


def test_regrid_unstructured_streams_through_dask():
    """End-to-end: regrid a chunked unstructured variable and
    confirm output has the F22.5 shape and finite values everywhere
    (xesmf nearest_s2d fills every target cell from the nearest
    source point — masking is applied downstream)."""
    from grid import make_target_grid

    ds = _unstructured_ds(ncells=500, ntime=4).chunk({"time": 2})
    target = make_target_grid("F22.5")
    regridder, method = make_regridder(ds, target, "bilinear")
    out = regridder(ds["tos"])
    # Streams as dask so the regrid graph composes without realizing.
    assert hasattr(out.data, "dask")
    arr = out.compute()
    assert arr.shape == (4, 45, 90)
    assert np.isfinite(arr.values).all()


# ---------------------------------------------------------------------------
# apply_target_land_mask
# ---------------------------------------------------------------------------


def test_apply_target_land_mask_nans_above_threshold():
    da = xr.DataArray(np.full((3, 4, 5), 290.0), dims=("time", "lat", "lon"))
    # Land fraction in [0, 1]; > 0.5 → land.
    land_fraction = xr.DataArray(
        np.where(np.arange(20).reshape(4, 5) >= 10, 1.0, 0.0),
        dims=("lat", "lon"),
    )
    masked = apply_target_land_mask(da, land_fraction)
    assert masked.shape == da.shape
    assert int(np.isnan(masked.isel(time=0).values).sum()) == 10
    assert float(masked.isel(time=0, lat=0, lon=0)) == 290.0


def test_apply_target_land_mask_threshold_respected():
    da = xr.DataArray(np.ones((2, 3)) * 5.0, dims=("lat", "lon"))
    land_fraction = xr.DataArray(
        [[0.10, 0.60, 0.90], [0.40, 0.49, 1.00]], dims=("lat", "lon")
    )
    masked = apply_target_land_mask(da, land_fraction)
    expected_nan = np.array([[False, True, True], [False, False, True]])
    np.testing.assert_array_equal(np.isnan(masked.values), expected_nan)


# ---------------------------------------------------------------------------
# apply_output_renames
# ---------------------------------------------------------------------------


def test_apply_output_renames_renames_and_tags():
    ds = xr.Dataset(
        {
            "rlds": (("lat", "lon"), np.full((2, 3), 250.0)),
            "rsds": (("lat", "lon"), np.full((2, 3), 200.0)),
            "tas": (("lat", "lon"), np.full((2, 3), 280.0)),
        }
    )
    out = apply_output_renames(ds, {"rlds": "DLWRFsfc", "rsds": "DSWRFsfc"})
    assert set(out.data_vars) == {"DLWRFsfc", "DSWRFsfc", "tas"}
    assert out["DLWRFsfc"].attrs["original_name"] == "rlds"
    assert out["DSWRFsfc"].attrs["original_name"] == "rsds"
    assert "original_name" not in out["tas"].attrs


def test_apply_output_renames_skips_absent_keys():
    ds = xr.Dataset({"pr": (("lat", "lon"), np.zeros((2, 3)))})
    out = apply_output_renames(ds, {"rlds": "DLWRFsfc", "pr": "PRATEsfc"})
    assert "PRATEsfc" in out.data_vars
    assert "DLWRFsfc" not in out.data_vars


# ---------------------------------------------------------------------------
# compute_ocean_fraction
# ---------------------------------------------------------------------------


def test_compute_ocean_fraction_basic():
    land = xr.DataArray(np.array([[0.0, 0.5, 1.0]]), dims=("lat", "lon"))
    ice = xr.DataArray(np.array([[0.2, 0.1, 0.0]]), dims=("lat", "lon"))
    ocean = compute_ocean_fraction(land, ice, "simon_ocean_fraction")
    np.testing.assert_allclose(ocean.values, [[0.8, 0.4, 0.0]])
    assert ocean.name == "simon_ocean_fraction"
    assert ocean.attrs["original_name"] == "derived"


def test_compute_ocean_fraction_clips_to_unit_interval():
    # land + ice > 1 (coastal sliver case) → ocean clipped to 0
    land = xr.DataArray(np.array([0.7]), dims=("lat",))
    ice = xr.DataArray(np.array([0.6]), dims=("lat",))
    ocean = compute_ocean_fraction(land, ice, "x")
    np.testing.assert_array_equal(ocean.values, [0.0])


def test_compute_ocean_fraction_broadcasts_time():
    """Static land + time-varying sea-ice should give time-varying ocean."""
    land = xr.DataArray(np.array([[0.0, 0.0]]), dims=("lat", "lon"))
    ice = xr.DataArray(
        np.array([[[0.3, 0.5]], [[0.6, 0.4]]]),
        dims=("time", "lat", "lon"),
    )
    ocean = compute_ocean_fraction(land, ice, "siday_ocean_fraction")
    assert ocean.dims == ("time", "lat", "lon")
    np.testing.assert_allclose(ocean.isel(time=0).values, [[0.7, 0.5]])
    np.testing.assert_allclose(ocean.isel(time=1).values, [[0.4, 0.6]])


# ---------------------------------------------------------------------------
# finalize_surface_and_ocean_variable — unit_scale + original_name attr
# ---------------------------------------------------------------------------


def test_finalize_surface_and_ocean_applies_unit_scale():
    """``simon_sea_ice_fraction`` should be scaled by 0.01 (%-→fraction)."""
    import cftime
    from config import SURFACE_AND_OCEAN_BY_OUTPUT

    h = SURFACE_AND_OCEAN_BY_OUTPUT["simon_sea_ice_fraction"]
    assert h.unit_scale == 0.01
    monthly_times = [cftime.DatetimeNoLeap(2010, m, 15) for m in range(1, 4)]
    # CMIP6 siconc is in [0, 100]; 75% over the ocean cells, NaN over land.
    arr = np.full((3, 4, 8), 75.0)
    arr[:, :2, :] = np.nan  # land cells
    monthly = xr.DataArray(
        arr, dims=("time", "lat", "lon"), coords={"time": monthly_times}
    )
    daily_times = xr.date_range(
        "2010-03-01", "2010-03-05", freq="D", use_cftime=True, calendar="noleap"
    )
    outputs = finalize_surface_and_ocean_variable(
        monthly, h, xr.DataArray(daily_times, dims=("time",))
    )
    # Output should be the previous-month value (Feb = 75) scaled to 0.75.
    da = outputs["simon_sea_ice_fraction"]
    assert float(da.where(~np.isnan(da)).max()) <= 1.0 + 1e-6
    np.testing.assert_allclose(float(da.where(~np.isnan(da)).mean()), 0.75)
    assert da.attrs["original_name"] == "siconc"


def test_finalize_surface_and_ocean_atmos_kind_carries_original_name():
    from config import SURFACE_AND_OCEAN_BY_OUTPUT

    h = SURFACE_AND_OCEAN_BY_OUTPUT["eday_ts"]
    daily_times = xr.date_range(
        "2010-01-01", "2010-01-05", freq="D", use_cftime=True, calendar="noleap"
    )
    da = xr.DataArray(
        np.full((5, 4, 8), 280.0),
        dims=("time", "lat", "lon"),
        coords={"time": daily_times},
    )
    outputs = finalize_surface_and_ocean_variable(
        da, h, xr.DataArray(daily_times, dims=("time",))
    )
    assert outputs["eday_ts"].attrs["original_name"] == "ts"


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok {name}")
            except Exception as e:
                print(f"FAIL {name}: {e}")
                failed += 1
    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    print("\nall tests passed")
