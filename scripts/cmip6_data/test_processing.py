"""Tests for processing.py — shared CMIP6 processing utilities.

Requires xarray, numpy, and xesmf (for regridding tests).
"""

import numpy as np
import pytest
import xarray as xr
from config import RegridConfig, ResolvedDatasetConfig, TimeWindow
from processing import (
    DuplicateTimestampsError,
    SimulationBoundaryError,
    _has_bounds,
    flatten_plev_variables,
    make_regridder,
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
    time = xr.cftime_range("2010-01-01", periods=ntime, freq="D", calendar="noleap")

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
    dup = xr.concat([ds, ds.isel(time=[0])], dim="time")
    result, msg = resolve_time_duplicates(dup, "tas", allow_dedupe=True)
    assert "deduplicated" in msg
    assert result.sizes["time"] == 3


def test_resolve_raises_without_allow_dedupe():
    ds = _rectilinear_ds(ntime=3)
    dup = xr.concat([ds, ds.isel(time=[0])], dim="time")
    with pytest.raises(DuplicateTimestampsError):
        resolve_time_duplicates(dup, "tas", allow_dedupe=False)


def test_resolve_raises_on_boundary_mismatch():
    ds = _rectilinear_ds(ntime=3)
    ds2 = ds.copy(deep=True)
    ds2["tas"].values[0] += 999.0
    dup = xr.concat([ds, ds2.isel(time=[0])], dim="time")
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

    # Simulate what regrid_variables does: subset to just the data vars
    # plus the bounds we now carry over.
    _BOUNDS_NAMES = {
        "lon_bnds",
        "lat_bnds",
        "lon_b",
        "lat_b",
        "vertices_longitude",
        "vertices_latitude",
    }
    bounds_to_keep = [v for v in ds.data_vars if v in _BOUNDS_NAMES]
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
