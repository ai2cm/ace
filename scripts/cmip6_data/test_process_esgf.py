"""Tests for process_esgf.py — ESGF pipeline orchestration."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from config import ESGFProcessConfig
from process_esgf import _open_netcdf_files, select_esgf_datasets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_netcdf(
    path: Path,
    variable: str,
    *,
    ntime: int = 5,
    nlat: int = 4,
    nlon: int = 8,
    with_bounds: bool = True,
    with_plev: bool = False,
) -> None:
    lat = np.linspace(-60, 60, nlat)
    lon = np.linspace(0, 315, nlon)
    time = xr.cftime_range("2010-01-01", periods=ntime, freq="D", calendar="noleap")

    coords: dict = {"lat": lat, "lon": lon, "time": time}
    dims = ["time", "lat", "lon"]
    if with_plev:
        plev = np.array([100000, 50000, 10000], dtype=np.float64)
        coords["plev"] = plev
        dims = ["time", "plev", "lat", "lon"]

    shape = tuple(len(coords[d]) for d in dims)
    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape).astype(np.float32)
    ds = xr.Dataset({variable: xr.DataArray(data, dims=dims)}, coords=coords)

    if with_bounds:
        dlat = np.diff(lat).mean()
        lat_bnds = np.column_stack([lat - dlat / 2, lat + dlat / 2])
        lat_bnds = np.clip(lat_bnds, -90, 90)
        dlon = np.diff(lon).mean()
        lon_bnds = np.column_stack([lon - dlon / 2, lon + dlon / 2])
        ds["lat_bnds"] = xr.DataArray(lat_bnds, dims=("lat", "bnds"))
        ds["lon_bnds"] = xr.DataArray(lon_bnds, dims=("lon", "bnds"))

    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# _open_netcdf_files
# ---------------------------------------------------------------------------


def test_open_netcdf_files_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "pr_day_2010.nc"
        _write_netcdf(p, "pr", ntime=5)
        ds = _open_netcdf_files([p], "pr")
        assert "pr" in ds.data_vars
        assert ds.sizes["time"] == 5


def test_open_netcdf_files_preserves_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "pr_day_2010.nc"
        _write_netcdf(p, "pr", ntime=3, with_bounds=True)
        ds = _open_netcdf_files([p], "pr")
        assert (
            "lat_bnds" in ds.data_vars
        ), "lat_bnds must survive for conservative regrid"
        assert (
            "lon_bnds" in ds.data_vars
        ), "lon_bnds must survive for conservative regrid"


def test_open_netcdf_files_no_bounds_no_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "ua_day_2010.nc"
        _write_netcdf(p, "ua", ntime=3, with_bounds=False)
        ds = _open_netcdf_files([p], "ua")
        assert "ua" in ds.data_vars
        assert "lat_bnds" not in ds.data_vars


def test_open_netcdf_files_concat_multiple():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, start in enumerate(["2010-01-01", "2010-01-06"]):
            p = Path(tmpdir) / f"tas_day_{i}.nc"
            lat = np.linspace(-60, 60, 4)
            lon = np.linspace(0, 315, 8)
            time = xr.cftime_range(start, periods=5, freq="D", calendar="noleap")
            data = np.zeros((5, 4, 8), dtype=np.float32)
            ds = xr.Dataset(
                {"tas": xr.DataArray(data, dims=["time", "lat", "lon"])},
                coords={"lat": lat, "lon": lon, "time": time},
            )
            ds.to_netcdf(p)
        ds = _open_netcdf_files(list(Path(tmpdir).glob("*.nc")), "tas")
        assert ds.sizes["time"] == 10


def test_open_netcdf_files_skips_files_without_variable():
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "pr_day_2010.nc"
        _write_netcdf(p1, "pr", ntime=3)
        p2 = Path(tmpdir) / "ua_day_2010.nc"
        _write_netcdf(p2, "ua", ntime=3)
        ds = _open_netcdf_files([p1, p2], "pr")
        assert "pr" in ds.data_vars
        assert ds.sizes["time"] == 3


def test_open_netcdf_files_raises_for_missing_variable():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "ua_day_2010.nc"
        _write_netcdf(p, "ua", ntime=3)
        with pytest.raises(ValueError, match="No data found"):
            _open_netcdf_files([p], "nonexistent")


def test_open_netcdf_files_with_plev():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "ua_day_2010.nc"
        _write_netcdf(p, "ua", ntime=3, with_plev=True)
        ds = _open_netcdf_files([p], "ua")
        assert "plev" in ds.dims
        assert ds.sizes["plev"] == 3


# ---------------------------------------------------------------------------
# select_esgf_datasets
# ---------------------------------------------------------------------------


def _make_inventory(
    models: list[dict],
) -> pd.DataFrame:
    """Build a minimal ESGF inventory DataFrame from per-model specs.

    Each entry in ``models`` should have at least ``source_id`` and
    ``variables`` (list of (table_id, variable_id) tuples). Optional
    keys: ``experiment_id``, ``member_id``, ``grid_label``, ``variant_r/i/p/f``.
    """
    rows = []
    for m in models:
        src = m["source_id"]
        exp = m.get("experiment_id", "historical")
        member = m.get("member_id", "r1i1p1f1")
        gl = m.get("grid_label", "gn")
        r = m.get("variant_r", 1)
        i = m.get("variant_i", 1)
        p = m.get("variant_p", 1)
        f = m.get("variant_f", 1)
        for table, var in m["variables"]:
            rows.append(
                {
                    "source_id": src,
                    "experiment_id": exp,
                    "member_id": member,
                    "table_id": table,
                    "variable_id": var,
                    "grid_label": gl,
                    "variant_r": r,
                    "variant_i": i,
                    "variant_p": p,
                    "variant_f": f,
                }
            )
    return pd.DataFrame(rows)


_CORE_VARS = [("day", v) for v in ["ua", "va", "hus", "zg", "tas", "huss", "psl", "pr"]]


def _minimal_config() -> ESGFProcessConfig:
    return ESGFProcessConfig(output_directory="/tmp/test")


def test_select_model_with_all_core_vars():
    inv = _make_inventory([{"source_id": "ModelA", "variables": _CORE_VARS}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 1
    assert tasks[0].source_id == "ModelA"


def test_select_skips_model_missing_core_var():
    partial = [v for v in _CORE_VARS if v[1] != "zg"]
    inv = _make_inventory([{"source_id": "ModelA", "variables": partial}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 0


def test_select_detects_forcings_and_statics():
    full_vars = _CORE_VARS + [
        ("Amon", "ts"),
        ("SImon", "siconc"),
        ("fx", "orog"),
        ("fx", "sftlf"),
    ]
    inv = _make_inventory([{"source_id": "ModelA", "variables": full_vars}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.has_ts is True
    assert t.has_siconc is True
    assert t.has_orog is True
    assert t.has_sftlf is True


def test_select_forcings_false_when_absent():
    inv = _make_inventory([{"source_id": "ModelA", "variables": _CORE_VARS}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    t = tasks[0]
    assert t.has_ts is False
    assert t.has_siconc is False
    assert t.has_orog is False
    assert t.has_sftlf is False


def test_select_respects_exclude_source_ids():
    inv = _make_inventory(
        [
            {"source_id": "ModelA", "variables": _CORE_VARS},
            {"source_id": "ModelB", "variables": _CORE_VARS},
        ]
    )
    cfg = _minimal_config()
    cfg.selection.exclude_source_ids = ["ModelB"]
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 1
    assert tasks[0].source_id == "ModelA"


def test_select_respects_source_ids_filter():
    inv = _make_inventory(
        [
            {"source_id": "ModelA", "variables": _CORE_VARS},
            {"source_id": "ModelB", "variables": _CORE_VARS},
        ]
    )
    cfg = _minimal_config()
    cfg.selection.source_ids = ["ModelB"]
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 1
    assert tasks[0].source_id == "ModelB"


def test_select_respects_experiment_filter():
    inv = _make_inventory(
        [
            {"source_id": "M", "experiment_id": "historical", "variables": _CORE_VARS},
            {"source_id": "M", "experiment_id": "ssp245", "variables": _CORE_VARS},
            {"source_id": "M", "experiment_id": "piControl", "variables": _CORE_VARS},
        ]
    )
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    sources = {t.experiment for t in tasks}
    assert "historical" in sources
    assert "ssp245" in sources
    assert "piControl" not in sources


def test_select_max_members_per_f():
    models = []
    for r in range(1, 6):
        models.append(
            {
                "source_id": "M",
                "experiment_id": "historical",
                "member_id": f"r{r}i1p1f1",
                "variant_r": r,
                "variables": _CORE_VARS,
            }
        )
    inv = _make_inventory(models)
    cfg = _minimal_config()
    cfg.selection.max_members_per_f = 3
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 3
    rs = sorted(t.variant_r for t in tasks)
    assert rs == [1, 2, 3]


def test_select_includes_optional_vars_in_available():
    extra_vars = _CORE_VARS + [("day", "sfcWind"), ("day", "hfls")]
    inv = _make_inventory([{"source_id": "M", "variables": extra_vars}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    assert "sfcWind" in tasks[0].available_day_variables
    assert "hfls" in tasks[0].available_day_variables


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
