"""Tests for process_esgf.py — ESGF pipeline orchestration."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import process_esgf
import pytest
import xarray as xr
from config import ESGFProcessConfig
from process_esgf import (
    _download_files,
    _open_netcdf_files,
    _select_day_augmentables,
    select_esgf_datasets,
)

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
    time = xr.date_range(
        "2010-01-01", periods=ntime, freq="D", calendar="noleap", use_cftime=True
    )

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
        assert "lat_bnds" in ds.data_vars, (
            "lat_bnds must survive for conservative regrid"
        )
        assert "lon_bnds" in ds.data_vars, (
            "lon_bnds must survive for conservative regrid"
        )


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
            time = xr.date_range(
                start, periods=5, freq="D", calendar="noleap", use_cftime=True
            )
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


def test_open_netcdf_files_overlapping_times():
    """Files with overlapping time ranges concat with duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lat = np.linspace(-60, 60, 4)
        lon = np.linspace(0, 315, 8)
        for i, (start, n) in enumerate([("2010-01-01", 10), ("2010-01-08", 10)]):
            p = Path(tmpdir) / f"tas_day_{i}.nc"
            time = xr.date_range(
                start, periods=n, freq="D", calendar="noleap", use_cftime=True
            )
            data = np.ones((n, 4, 8), dtype=np.float32) * (i + 1)
            ds = xr.Dataset(
                {"tas": xr.DataArray(data, dims=["time", "lat", "lon"])},
                coords={"lat": lat, "lon": lon, "time": time},
            )
            ds.to_netcdf(p)
        ds = _open_netcdf_files(list(Path(tmpdir).glob("*.nc")), "tas")
        assert ds.sizes["time"] == 20


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


def test_select_skips_model_missing_core_var_with_strict_gate():
    partial = [v for v in _CORE_VARS if v[1] != "zg"]
    inv = _make_inventory([{"source_id": "ModelA", "variables": partial}])
    cfg = _minimal_config()
    cfg.defaults.max_core_missing = 0
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 0


def test_select_tolerates_one_missing_core_var_with_default_gate():
    """Default ``max_core_missing=3`` lets a model with just ``zg``
    missing through; the variable is simply absent from the output.
    """
    partial = [v for v in _CORE_VARS if v[1] != "zg"]
    inv = _make_inventory([{"source_id": "ModelA", "variables": partial}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 1


def test_select_skips_model_missing_too_many_core_vars():
    """A model missing more than ``max_core_missing`` is dropped."""
    partial = [v for v in _CORE_VARS if v[1] not in ("ua", "va", "hus", "zg")]
    inv = _make_inventory([{"source_id": "ModelA", "variables": partial}])
    cfg = _minimal_config()
    cfg.defaults.max_core_missing = 3  # default; 4 missing > 3
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 0


def test_select_detects_surface_and_ocean_and_statics():
    full_vars = _CORE_VARS + [
        ("Amon", "ts"),
        ("SImon", "siconc"),
        ("Eday", "ts"),
        ("Oday", "tos"),
        ("fx", "orog"),
        ("fx", "sftlf"),
    ]
    inv = _make_inventory([{"source_id": "ModelA", "variables": full_vars}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    assert len(tasks) == 1
    t = tasks[0]
    assert "amon_ts" in t.available_surface_and_ocean_variables
    assert "simon_sea_ice_fraction" in t.available_surface_and_ocean_variables
    assert "surface_temperature" in t.available_surface_and_ocean_variables
    assert "oday_tos" in t.available_surface_and_ocean_variables
    assert t.has_orog is True
    assert t.has_sftlf is True


def test_select_surface_and_ocean_absent_when_only_core():
    inv = _make_inventory([{"source_id": "ModelA", "variables": _CORE_VARS}])
    cfg = _minimal_config()
    tasks = select_esgf_datasets(inv, cfg)
    t = tasks[0]
    assert t.available_surface_and_ocean_variables == []
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


def test_select_respects_exclude_variants():
    """``exclude_variants`` drops the named (source, exp, variant) triple
    while leaving sibling variants of the same model alone."""
    from config import VariantKey

    inv = _make_inventory(
        [
            {
                "source_id": "ModelA",
                "variables": _CORE_VARS,
                "member_id": "r1i1p1f1",
                "variant_r": 1,
            },
            {
                "source_id": "ModelA",
                "variables": _CORE_VARS,
                "member_id": "r2i1p1f1",
                "variant_r": 2,
            },
            {
                "source_id": "ModelA",
                "variables": _CORE_VARS,
                "member_id": "r4i1p1f1",
                "variant_r": 4,
            },
        ]
    )
    cfg = _minimal_config()
    cfg.selection.exclude_variants = [
        VariantKey(
            source_id="ModelA",
            experiment="historical",
            variant_label="r4i1p1f1",
        )
    ]
    tasks = select_esgf_datasets(inv, cfg)
    kept = sorted(t.variant_label for t in tasks)
    assert kept == ["r1i1p1f1", "r2i1p1f1"]


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


# ---------------------------------------------------------------------------
# _select_day_augmentables — pure-function filter for the day-cadence
# augmenter (downloads + writes are integration-only and not exercised
# here; this just locks the filter logic).
# ---------------------------------------------------------------------------


def test_day_augmentables_picks_published_and_missing_from_existing():
    """Variables are augmentable iff published per
    ``available_day_variables`` AND their renamed output name is
    missing from the existing zarr."""
    # ``rsdt`` (CFday) renames to ``DSWRFtoa``; ``rsds`` (day)
    # renames to ``DSWRFsfc``; ``wap500`` keeps its name.
    out = _select_day_augmentables(
        optional_variables=["rsdt", "rsds", "wap500"],
        available_day_variables=["rsdt", "rsds", "wap500", "tas"],
        existing_vars={"DSWRFsfc"},  # rsds already in
    )
    assert out == ["rsdt", "wap500"]  # rsds excluded, others kept


def test_day_augmentables_skips_unpublished_vars():
    """A variable not in ``available_day_variables`` is not
    augmentable even if absent from the existing zarr — the source
    model doesn't publish it."""
    out = _select_day_augmentables(
        optional_variables=["rsdt", "rsds", "wap500"],
        available_day_variables=["rsds"],  # only rsds is published
        existing_vars=set(),
    )
    assert out == ["rsds"]


def test_day_augmentables_respects_rename_map_for_existing_check():
    """The existing-zarr filter checks the RENAMED output name, not
    the CMIP6 source name. ``rsdt`` written to the zarr as
    ``DSWRFtoa`` should still block its own re-augmentation."""
    out = _select_day_augmentables(
        optional_variables=["rsdt", "rsut"],
        available_day_variables=["rsdt", "rsut"],
        existing_vars={"DSWRFtoa"},  # rsdt's output name
    )
    assert out == ["rsut"]


def test_day_augmentables_no_rename_variable():
    """Variables without an entry in ``CMIP_TO_OUTPUT_RENAMES``
    (e.g. ``wap500``, ``clivi``, ``clwvi``) keep their CMIP6 name
    as the output name; the existing-zarr filter uses that name
    directly."""
    out = _select_day_augmentables(
        optional_variables=["wap500", "clivi", "clwvi"],
        available_day_variables=["wap500", "clivi", "clwvi"],
        existing_vars={"clivi"},
    )
    assert out == ["wap500", "clwvi"]


def test_day_augmentables_returns_empty_when_nothing_to_do():
    out = _select_day_augmentables(
        optional_variables=["rsdt", "rsut"],
        available_day_variables=["rsdt", "rsut"],
        existing_vars={"DSWRFtoa", "USWRFtoa"},  # both already present
    )
    assert out == []


def test_day_augmentables_skips_previously_failed():
    """A variable a prior pass tried + failed (recorded in the sidecar
    as ``esgf_failed_augment_variables``) must not be re-attempted —
    that's the whole point of persisting the failure state. The match
    is on the renamed output name (same convention as
    ``existing_vars``)."""
    # ``rsdt`` renames to ``DSWRFtoa``; if a prior pass recorded
    # ``DSWRFtoa`` as failed, the next pass should skip it even
    # though it's published and not already in the zarr.
    out = _select_day_augmentables(
        optional_variables=["rsdt", "rsut"],
        available_day_variables=["rsdt", "rsut"],
        existing_vars=set(),
        failed_augment_vars={"DSWRFtoa"},
    )
    assert out == ["rsut"]


def test_day_augmentables_failed_default_is_empty_set():
    """The ``failed_augment_vars`` parameter defaults to None and is
    treated as the empty set — callers that don't track persisted
    failure state (e.g. older callers, unit tests) keep working."""
    out = _select_day_augmentables(
        optional_variables=["rsdt"],
        available_day_variables=["rsdt"],
        existing_vars=set(),
    )
    assert out == ["rsdt"]


# ---------------------------------------------------------------------------
# _should_derive_total_water_path — predicate for the derived TWP write
# in augment_one_esgf. Tests the same-pass-both-inputs case that the
# original implementation missed (22 of 26 eligible v2 datasets).
# ---------------------------------------------------------------------------


def test_total_water_path_derived_when_both_added_in_same_pass():
    """The common v2 case: Pangeo ingest had neither water_vapor_path
    nor clwvi; this augment pass added water_vapor_path via the
    surface-and-ocean loop and clwvi via the day-cadence loop.
    Pre-fix this returned False because water_vapor_path wasn't in
    ``existing_vars``."""
    from process_esgf import _should_derive_total_water_path

    assert _should_derive_total_water_path(
        day_added=["clwvi"],
        existing_vars=set(),
        added_names=["water_vapor_path"],
    )


def test_total_water_path_derived_when_wvp_preexisting_and_clwvi_added():
    """The 4-of-26 case that the pre-fix code already handled:
    water_vapor_path was in the Pangeo zarr already and clwvi got
    augmented in this pass."""
    from process_esgf import _should_derive_total_water_path

    assert _should_derive_total_water_path(
        day_added=["clwvi"],
        existing_vars={"water_vapor_path"},
        added_names=[],
    )


def test_total_water_path_skipped_when_clwvi_already_present():
    """If clwvi was added in a *prior* augment pass (not this one),
    skip the derivation. Either total_water_path already exists from
    that prior pass, or it was deliberately not derived then. Don't
    second-guess history; force-rederive via a migration if needed."""
    from process_esgf import _should_derive_total_water_path

    assert not _should_derive_total_water_path(
        day_added=[],
        existing_vars={"water_vapor_path", "clwvi"},
        added_names=[],
    )


def test_total_water_path_skipped_when_wvp_missing_everywhere():
    """clwvi alone is not enough — without water_vapor_path the
    derivation is undefined."""
    from process_esgf import _should_derive_total_water_path

    assert not _should_derive_total_water_path(
        day_added=["clwvi"],
        existing_vars=set(),
        added_names=[],
    )


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


# ---------------------------------------------------------------------------
# Parallel downloads
# ---------------------------------------------------------------------------


class _FakeFile:
    def __init__(self, name):
        self.filename = name


def test_download_files_preserves_order_parallel(monkeypatch):
    """Concurrent downloads must return paths in input (time) order even when
    later files finish first, and call download_file once per file."""
    import time as _time

    calls = []

    def fake_download(f, scratch):
        # finish in reverse order so a naive as-completed gather would scramble
        idx = int(f.filename.split("_")[1].split(".")[0])
        _time.sleep(0.02 * (5 - idx))
        calls.append(f.filename)
        return Path(scratch) / f.filename

    monkeypatch.setattr(process_esgf, "download_file", fake_download)
    files = [_FakeFile(f"f_{i}.nc") for i in range(5)]
    out = _download_files(files, Path("/scratch"), n_workers=4)
    assert [p.name for p in out] == [f"f_{i}.nc" for i in range(5)]
    assert sorted(calls) == sorted(f.filename for f in files)


def test_download_files_serial_path(monkeypatch):
    """n_workers=1 still downloads every file, in order."""
    monkeypatch.setattr(
        process_esgf, "download_file", lambda f, scratch: Path(scratch) / f.filename
    )
    files = [_FakeFile(f"f_{i}.nc") for i in range(3)]
    out = _download_files(files, Path("/scratch"), n_workers=1)
    assert [p.name for p in out] == ["f_0.nc", "f_1.nc", "f_2.nc"]


def test_download_files_propagates_failure(monkeypatch):
    """A download failure propagates (matching the old serial loop)."""

    def fake_download(f, scratch):
        if f.filename == "f_1.nc":
            raise RuntimeError("data node down")
        return Path(scratch) / f.filename

    monkeypatch.setattr(process_esgf, "download_file", fake_download)
    files = [_FakeFile(f"f_{i}.nc") for i in range(3)]
    with pytest.raises(RuntimeError, match="data node down"):
        _download_files(files, Path("/scratch"), n_workers=4)


# ---------------------------------------------------------------------------
# _masked_regrid_3d_state — source-grid below-surface masking orchestration.
# Verified with a fake block-mean regridder + mocked ESGF, no ESMF/network.
# ---------------------------------------------------------------------------


class _BlockMeanRegridder:
    """Fake xESMF regridder: coarsen each 2x2 lat/lon block by mean.

    Accepts the same call kwargs the real regridder is invoked with
    (``keep_attrs``, ``skipna``) so the injected seam is signature-compatible.
    """

    def __call__(self, da, keep_attrs=False, skipna=True):
        return da.coarsen(lat=2, lon=2).mean(skipna=skipna)


def _fake_make_regridder(source_ds, target, method):
    return _BlockMeanRegridder(), method


def _native_3d_ds(name, plev, blocks_by_plev, ntime=2):
    """Native (time, plev, lat, lon) single-variable dataset from per-plev 2D
    arrays, broadcast across ``ntime`` timesteps."""
    per_plev = [np.asarray(b, dtype="float64") for b in blocks_by_plev]
    nlat, nlon = per_plev[0].shape
    arr = np.broadcast_to(
        np.stack(per_plev)[None], (ntime, len(plev), nlat, nlon)
    ).copy()
    time = xr.date_range(
        "2010-01-01", periods=ntime, freq="D", calendar="noleap", use_cftime=True
    )
    return xr.Dataset(
        {name: (("time", "plev", "lat", "lon"), arr)},
        coords={
            "time": time,
            "plev": plev,
            "lat": np.arange(nlat),
            "lon": np.arange(nlon),
        },
    )


def _orog_ds(values):
    arr = np.asarray(values, dtype="float64")
    nlat, nlon = arr.shape
    return xr.Dataset(
        {"orog": (("lat", "lon"), arr)},
        coords={"lat": np.arange(nlat), "lon": np.arange(nlon)},
    )


def _masking_task():
    return process_esgf.ESGFDatasetTask(
        source_id="MPI-ESM1-2-LR",
        experiment="historical",
        variant_label="r1i1p1f1",
        variant_r=1,
        variant_i=1,
        variant_p=1,
        variant_f=1,
        grid_label="gn",
        available_day_variables=["zg", "ua"],
        has_orog=True,
    )


def test_masked_regrid_3d_state_shared_mask_and_cleanup(monkeypatch):
    """Native download seam mocked; a single native zg<orog test drives the
    shared per-level mask for every field, and native files are cleaned up."""
    plev = np.array([100000.0, 85000.0])  # 1000, 850 hPa (descending pressure)
    # 1000 hPa: top-left native cell at 100 m (below the 500 m surface); rest
    # above. 850 hPa: fully above surface.
    zg1000 = np.full((4, 4), 1000.0)
    zg1000[0, 0] = 100.0
    natives = {
        "zg": _native_3d_ds("zg", plev, [zg1000, np.full((4, 4), 3000.0)]),
        "ua": _native_3d_ds(
            "ua", plev, [np.arange(16.0).reshape(4, 4), np.full((4, 4), 7.0)]
        ),
        "orog": _orog_ds(np.full((4, 4), 500.0)),
    }

    def fake_download_native(task, variable, table_id, config, scratch):
        return natives.get(variable)

    cleaned: list[str] = []
    monkeypatch.setattr(process_esgf, "_download_native_variable", fake_download_native)
    monkeypatch.setattr(
        process_esgf,
        "cleanup_variable_files",
        lambda scratch, v: cleaned.append(v),
    )

    regridded, valid, hgtsfc, methods = process_esgf._masked_regrid_3d_state(
        _masking_task(),
        _minimal_config(),
        target_grid=xr.Dataset(),  # ignored by the fake regridder
        scratch=Path("/scratch"),
        present_3d=["zg", "ua"],
        make_regridder_fn=_fake_make_regridder,
        threshold=0.5,
    )

    # Shared target-grid per-level validity: top-left 1000 hPa cell had 3/4
    # coverage (>=0.5 -> valid); everything else valid.
    assert valid.dims == ("time", "plev", "lat", "lon")
    assert valid.sel(plev=100000.0).values[0].tolist() == [[True, True], [True, True]]
    assert bool(valid.sel(plev=85000.0).all())
    # ua's masked-regridded top-left 1000 hPa cell averages only the 3 valid
    # native cells (1, 4, 5), never the below-surface 0.
    np.testing.assert_allclose(
        regridded["ua"].sel(plev=100000.0).values[0, 0, 0], (1 + 4 + 5) / 3
    )
    assert set(regridded) == {"zg", "ua"}
    assert methods == {"zg": "bilinear", "ua": "bilinear"}
    assert hgtsfc.dims == ("lat", "lon")  # regridded surface height
    # every downloaded native (orog + zg + ua) is cleaned up
    assert set(cleaned) == {"orog", "zg", "ua"}


def test_masked_regrid_3d_state_bails_without_orog(monkeypatch):
    """No orography -> can't build the source-grid mask -> signal fallback."""
    monkeypatch.setattr(process_esgf, "_download_native_variable", lambda *a, **k: None)
    task = _masking_task()
    task.has_orog = False
    out = process_esgf._masked_regrid_3d_state(
        task,
        _minimal_config(),
        target_grid=xr.Dataset(),
        scratch=Path("/scratch"),
        present_3d=["zg", "ua"],
        make_regridder_fn=_fake_make_regridder,
    )
    assert out == (None, None, None, {})


def test_masked_regrid_3d_state_full_network_mock(monkeypatch, tmp_path):
    """End-to-end through the true network seam (query_files/download_file):
    idealized native netCDFs on disk, fake regridder, no ESMF."""
    plev = np.array([100000.0, 85000.0], dtype="float64")
    zg1000 = np.full((4, 4), 1000.0)
    zg1000[0, 0] = 100.0  # below the 500 m surface
    written = {
        "zg": _native_3d_ds("zg", plev, [zg1000, np.full((4, 4), 3000.0)]),
        "ua": _native_3d_ds(
            "ua", plev, [np.arange(16.0).reshape(4, 4), np.full((4, 4), 7.0)]
        ),
        "orog": _orog_ds(np.full((4, 4), 500.0)),
    }
    paths = {}
    for var, ds in written.items():
        p = tmp_path / f"{var}.nc"
        ds.to_netcdf(p)
        paths[var] = p

    class _Fileset:
        def __init__(self, var):
            self.files = [_FakeFile(var)]  # .filename carries the variable

        @property
        def total_size(self):
            return 1

    def fake_query(node, source, exp, member, table, variable):
        return _Fileset(variable)

    monkeypatch.setattr(process_esgf, "query_files", fake_query)
    monkeypatch.setattr(process_esgf, "filter_files_by_time", lambda fs, s, e: fs)
    monkeypatch.setattr(
        process_esgf, "download_file", lambda f, scratch: paths[f.filename]
    )

    regridded, valid, hgtsfc, methods = process_esgf._masked_regrid_3d_state(
        _masking_task(),
        _minimal_config(),
        target_grid=xr.Dataset(),
        scratch=tmp_path,
        present_3d=["zg", "ua"],
        make_regridder_fn=_fake_make_regridder,
        threshold=0.5,
    )
    assert set(regridded) == {"zg", "ua"}
    assert valid.sel(plev=100000.0).values[0].tolist() == [[True, True], [True, True]]
    np.testing.assert_allclose(
        regridded["ua"].sel(plev=100000.0).values[0, 0, 0], (1 + 4 + 5) / 3
    )
