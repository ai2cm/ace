"""Tests for processing.py — shared CMIP6 processing utilities.

Requires xarray, numpy, and xesmf (for regridding tests).
"""

import numpy as np
import pytest
import xarray as xr
from config import RegridConfig, ResolvedDatasetConfig, TimeWindow
from processing import (
    BOUNDS_NAMES,
    DuplicateTimestampsError,
    SimulationBoundaryError,
    _has_bounds,
    apply_time_subset,
    clip_date_for_calendar,
    compute_below_surface_mask,
    compute_derived_layer_T,
    fill_derived_layer_T,
    flatten_plev_variables,
    interp_monthly_to_daily,
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
    defaults = dict(
        source_id="TEST",
        experiment="historical",
        variant_label="r1i1p1f1",
        core_variables=["ua", "va", "hus", "zg"],
        optional_variables=["tas"],
        forcing_variables=["ts"],
        static_variables=["orog"],
        forcing_interpolation="linear",
        time_subset={"historical": TimeWindow("2010-01-01", "2010-12-31")},
        target_grid=type("TG", (), {"name": "F22.5"})(),
        regrid=RegridConfig(),
        fill=type("FC", (), {"method": "nearest_above"})(),
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
# interp_monthly_to_daily
# ---------------------------------------------------------------------------


def test_interp_monthly_to_daily_linear():
    monthly_time = xr.date_range("2010-01-15", periods=12, freq="MS", calendar="noleap")
    monthly = xr.DataArray(
        np.linspace(280, 300, 12).astype(np.float32),
        dims=["time"],
        coords={"time": monthly_time},
    )
    daily_time = xr.DataArray(
        xr.date_range("2010-01-01", periods=365, freq="D", calendar="noleap"),
        dims=["time"],
    )
    result = interp_monthly_to_daily(monthly, daily_time, "linear")
    assert result.sizes["time"] == 365
    assert not np.isnan(result.values).any()


def test_interp_monthly_to_daily_extrapolates_edges():
    monthly_time = xr.date_range("2010-02-15", periods=2, freq="MS", calendar="noleap")
    monthly = xr.DataArray(
        np.array([280.0, 290.0], dtype=np.float32),
        dims=["time"],
        coords={"time": monthly_time},
    )
    daily_time = xr.DataArray(
        xr.date_range("2010-01-01", periods=120, freq="D", calendar="noleap"),
        dims=["time"],
    )
    result = interp_monthly_to_daily(monthly, daily_time, "linear")
    assert not np.isnan(result.values).any()
    assert float(result.isel(time=0)) == 280.0


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
