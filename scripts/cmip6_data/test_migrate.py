"""Tests for the schema-migration framework.

Three layers:

1. **Unit** — version parsing/comparison, chain composition, sidecar
   round-tripping, ``migrate_one`` dispatching (no real zarrs touched).
2. **Migration-specific unit** — the 0.0.0 → 0.1.0 migration applied to
   a hand-crafted minimal "old schema" zarr (variables named
   ``eday_ts`` / ``orog``); verifies the rename, the bumped
   ``schema_version``, the regenerated ``stats.nc``, and the audit
   trail in the sidecar.
3. **Integration with the processing pipeline** — runs ``process_one``
   end-to-end against a synthetic mini-CMIP6 source (in-process,
   no GCS), then "ages" the output to the pre-rename schema (the
   inverse of migration 0.0.0 → 0.1.0) so the same migration code
   can roll it forward again. This catches drift between the
   pipeline's variable-naming and the migration assumptions.

The integration test is intentionally heavyweight; it's the safety
net for "we forgot to update the migration when we changed the
pipeline" — exactly what the user asked for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Layer 1: version utilities + chain composition
# ---------------------------------------------------------------------------


def test_parse_version_empty_is_zero():
    from schema_version import parse_version

    assert parse_version("") == (0, 0, 0)
    assert parse_version("0.0.0") == (0, 0, 0)


def test_parse_version_basic():
    from schema_version import parse_version

    assert parse_version("0.1.0") == (0, 1, 0)
    assert parse_version("12.34.5") == (12, 34, 5)


def test_parse_version_invalid_raises():
    from schema_version import parse_version

    with pytest.raises(ValueError):
        parse_version("v1.0.0")
    with pytest.raises(ValueError):
        parse_version("0.1")
    with pytest.raises(ValueError):
        parse_version("0.1.x")


def test_version_lt_strict_ordering():
    from schema_version import version_lt

    assert version_lt("0.0.0", "0.1.0")
    assert version_lt("0.1.0", "0.1.1")
    assert version_lt("0.1.0", "1.0.0")
    assert not version_lt("0.1.0", "0.1.0")
    assert not version_lt("0.2.0", "0.1.0")


def test_chain_for_empty_when_at_target():
    from migrations import chain_for
    from schema_version import SCHEMA_VERSION

    assert chain_for(SCHEMA_VERSION, SCHEMA_VERSION) == []


def test_chain_for_returns_registered_steps():
    from migrations import MIGRATIONS, chain_for
    from schema_version import SCHEMA_VERSION

    if not MIGRATIONS:
        pytest.skip("no migrations registered yet")
    chain = chain_for("0.0.0", SCHEMA_VERSION)
    # Chain composes: each step starts where the previous ended.
    assert chain[0].from_version == "0.0.0"
    for prev, curr in zip(chain, chain[1:]):
        assert prev.to_version == curr.from_version
    assert chain[-1].to_version == SCHEMA_VERSION


def test_chain_for_raises_on_unreachable_target():
    from migrations import chain_for

    with pytest.raises(RuntimeError, match="no migration registered"):
        chain_for("0.0.0", "99.99.99")


# ---------------------------------------------------------------------------
# Layer 1.5: migrate_one dispatch (no real zarr, just sidecar plumbing)
# ---------------------------------------------------------------------------


def _make_sidecar_only_dataset(tmp_path: Path, sidecar: dict) -> str:
    """Create a fake "zarr" directory with just a metadata.json sidecar.
    Useful for testing migrate_one's dispatch logic without standing up
    a real zarr.
    """
    zarr_dir = tmp_path / "data.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "metadata.json").write_text(json.dumps(sidecar))
    return str(zarr_dir)


def test_migrate_one_returns_current_when_already_at_target(tmp_path: Path):
    from migrate import migrate_one
    from schema_version import SCHEMA_VERSION

    zarr = _make_sidecar_only_dataset(
        tmp_path,
        {
            "source_id": "TEST",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
        },
    )
    status, _ = migrate_one(zarr)
    assert status == "current"


def test_migrate_one_missing_sidecar(tmp_path: Path):
    from migrate import migrate_one

    # Point at a directory with no metadata.json.
    empty = tmp_path / "data.zarr"
    empty.mkdir()
    status, _ = migrate_one(str(empty))
    assert status == "missing"


def test_migrate_one_dry_run_describes_chain(tmp_path: Path):
    from migrate import migrate_one
    from schema_version import SCHEMA_VERSION

    zarr = _make_sidecar_only_dataset(
        tmp_path,
        {
            "source_id": "TEST",
            "experiment": "historical",
            "variant_label": "r1i1p1f1",
            "status": "ok",
            # No schema_version → treated as 0.0.0.
        },
    )
    status, detail = migrate_one(zarr, target_version=SCHEMA_VERSION, dry_run=True)
    assert status == "would-migrate"
    assert "0.0.0" in detail
    assert SCHEMA_VERSION in detail
    # Dry-run must NOT mutate the sidecar.
    after = json.loads((Path(zarr) / "metadata.json").read_text())
    assert after.get("schema_version", "") == ""


# ---------------------------------------------------------------------------
# Layer 2: 0.0.0 → 0.1.0 migration on a minimal hand-crafted zarr
# ---------------------------------------------------------------------------


def _make_old_schema_zarr(zarr_path: Path) -> xr.Dataset:
    """Write a tiny dataset under the 0.0.0 ("eday_ts" / "orog") schema.
    Mimics the on-disk layout that ``process.py`` writes (chunks + shards
    via the F22.5 target shape, but small enough to be cheap in CI).
    """
    nt, nlat, nlon = 12, 45, 90
    times = xr.date_range("2010-01-01", periods=nt, freq="MS", calendar="noleap")
    rng = np.random.default_rng(0)
    eday_ts = xr.DataArray(
        280 + rng.standard_normal((nt, nlat, nlon)).astype(np.float32) * 5,
        dims=("time", "lat", "lon"),
        coords={"time": times},
        attrs={"original_name": "ts", "units": "K"},
    )
    orog = xr.DataArray(
        rng.uniform(0, 4000, (nlat, nlon)).astype(np.float32),
        dims=("lat", "lon"),
        attrs={"original_name": "orog", "units": "m"},
    )
    # A non-temperature, non-renamed variable to sanity-check the
    # migration leaves untouched.
    psl = xr.DataArray(
        100000 + rng.standard_normal((nt, nlat, nlon)).astype(np.float32) * 1000,
        dims=("time", "lat", "lon"),
        coords={"time": times},
        attrs={"units": "Pa"},
    )
    ds = xr.Dataset({"eday_ts": eday_ts, "orog": orog, "psl": psl})
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True, zarr_format=3)
    return ds


def _write_sidecar(zarr_dir: Path, sidecar: dict) -> None:
    (zarr_dir / "metadata.json").write_text(json.dumps(sidecar))


def test_migration_0_0_0_to_0_1_0_renames_variables(tmp_path: Path):
    from migrations._0_0_0_to_0_1_0 import MIGRATION

    zarr_dir = tmp_path / "data.zarr"
    src_ds = _make_old_schema_zarr(zarr_dir)
    sidecar = {
        "source_id": "TEST",
        "experiment": "historical",
        "variant_label": "r1i1p1f1",
        "label": "TEST",
        "status": "ok",
        "variables_present": sorted(src_ds.data_vars),
    }
    out_sidecar = MIGRATION.apply(str(zarr_dir), sidecar)

    # On-disk schema check.
    migrated = xr.open_zarr(str(zarr_dir), consolidated=True)
    assert "eday_ts" not in migrated.data_vars
    assert "orog" not in migrated.data_vars
    assert "surface_temperature" in migrated.data_vars
    assert "HGTsfc" in migrated.data_vars
    assert "psl" in migrated.data_vars  # untouched

    # Values preserved bit-for-bit (rename only).
    np.testing.assert_array_equal(
        migrated["surface_temperature"].values, src_ds["eday_ts"].values
    )
    np.testing.assert_array_equal(migrated["HGTsfc"].values, src_ds["orog"].values)

    # original_name preserved when it was already set.
    assert migrated["surface_temperature"].attrs.get("original_name") == "ts"
    assert migrated["HGTsfc"].attrs.get("original_name") == "orog"

    # Sidecar bumped + audited.
    assert out_sidecar["schema_version"] == "0.1.0"
    assert "surface_temperature" in out_sidecar["variables_present"]
    assert "HGTsfc" in out_sidecar["variables_present"]
    assert "eday_ts" not in out_sidecar["variables_present"]
    audit = out_sidecar["migrations"]
    assert audit[-1] == {
        "from": "0.0.0",
        "to": "0.1.0",
        "renamed": {"eday_ts": "surface_temperature", "orog": "HGTsfc"},
    }


def test_migration_0_0_0_to_0_1_0_noop_when_vars_absent(tmp_path: Path):
    """Dataset missing both eday_ts and orog should bump the version
    cleanly without rewriting the zarr."""
    from migrations._0_0_0_to_0_1_0 import MIGRATION

    zarr_dir = tmp_path / "data.zarr"
    nt, nlat, nlon = 5, 45, 90
    times = xr.date_range("2010-01-01", periods=nt, freq="D", calendar="noleap")
    ds = xr.Dataset(
        {
            "psl": xr.DataArray(
                np.full((nt, nlat, nlon), 1.0e5, dtype=np.float32),
                dims=("time", "lat", "lon"),
                coords={"time": times},
                attrs={"units": "Pa"},
            )
        }
    )
    ds.to_zarr(str(zarr_dir), mode="w", consolidated=True, zarr_format=3)
    sidecar = {"source_id": "T", "experiment": "h", "variant_label": "r1", "label": ""}
    out = MIGRATION.apply(str(zarr_dir), sidecar)
    assert out["schema_version"] == "0.1.0"
    assert out["migrations"][-1]["renamed"] == {}


def test_migration_0_0_0_to_0_1_0_regenerates_stats(tmp_path: Path):
    """Stats.nc must be rewritten with the new variable names —
    ``<var>__<stat>`` keys reference variable names directly."""
    from migrations._0_0_0_to_0_1_0 import MIGRATION

    zarr_dir = tmp_path / "data.zarr"
    _make_old_schema_zarr(zarr_dir)
    sidecar = {
        "source_id": "TEST",
        "experiment": "historical",
        "variant_label": "r1i1p1f1",
        "label": "TEST",
    }
    MIGRATION.apply(str(zarr_dir), sidecar)

    stats_path = tmp_path / "stats.nc"
    assert stats_path.exists(), "stats.nc should be regenerated after the rename"
    stats = xr.open_dataset(str(stats_path))
    keys = set(stats.data_vars)
    # New names appear; old names don't.
    assert any(k.startswith("surface_temperature__") for k in keys)
    assert any(k.startswith("HGTsfc__") for k in keys)
    assert not any(k.startswith("eday_ts__") for k in keys)
    assert not any(k.startswith("orog__") for k in keys)
    stats.close()


# ---------------------------------------------------------------------------
# Layer 3: integration with the processing pipeline
# ---------------------------------------------------------------------------


def _build_source_zstore(
    tmp_path: Path, variable_id: str, units: str = "", with_plev: bool = False
) -> str:
    """Write a tiny CMIP6-style source zarr (lat/lon/time + optional plev)
    that ``process.py``'s open / regrid path can read. Returns the local
    filesystem URL — ``_open_zstore`` supports any fsspec scheme.
    """
    nt, nlat, nlon = 6, 18, 36  # tiny native grid
    times = xr.date_range("2010-01-01", periods=nt, freq="D", calendar="noleap")
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(2.5, 357.5, nlon)
    rng = np.random.default_rng(hash(variable_id) & 0xFFFFFFFF)
    coords: dict = {"lat": lat, "lon": lon, "time": times}
    dims = ["time", "lat", "lon"]
    if with_plev:
        coords["plev"] = np.array(
            [100000, 85000, 70000, 50000, 25000, 10000, 5000, 1000], dtype=np.float64
        )
        dims = ["time", "plev", "lat", "lon"]
    shape = tuple(len(coords[d]) for d in dims)
    data = rng.standard_normal(shape).astype(np.float32)
    da = xr.DataArray(
        data, dims=dims, attrs={"cell_methods": "time: mean", "units": units or "1"}
    )
    ds = xr.Dataset({variable_id: da}, coords=coords)
    path = tmp_path / f"{variable_id}.zarr"
    ds.to_zarr(str(path), mode="w", consolidated=True, zarr_format=3)
    return str(path)


def _build_static_zstore(tmp_path: Path, var_id: str) -> str:
    """Write a CMIP6 fx-table style static zarr."""
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(2.5, 357.5, nlon)
    rng = np.random.default_rng(hash(var_id) & 0xFFFFFFFF)
    if var_id == "sftlf":
        data = (rng.uniform(0, 1, (nlat, nlon)) * 100).astype(np.float32)
    elif var_id == "orog":
        data = (rng.uniform(0, 3000, (nlat, nlon))).astype(np.float32)
    else:
        data = rng.standard_normal((nlat, nlon)).astype(np.float32)
    units = "%" if var_id == "sftlf" else "m"
    ds = xr.Dataset(
        {var_id: xr.DataArray(data, dims=("lat", "lon"), attrs={"units": units})},
        coords={"lat": lat, "lon": lon},
    )
    path = tmp_path / f"static_{var_id}.zarr"
    ds.to_zarr(str(path), mode="w", consolidated=True, zarr_format=3)
    return str(path)


@pytest.mark.slow
def test_integration_pipeline_then_migration(tmp_path: Path):
    """End-to-end: build a synthetic mini CMIP6 source tree, run
    ``process_one`` to produce a current-schema dataset, manually
    invert the 0.0.0 → 0.1.0 rename to simulate an older write, then
    apply the migration and verify the result matches what
    ``process_one`` produced directly.

    This is the regression guard the user asked for: it catches the
    "we updated the pipeline but forgot to update the migration"
    failure mode by forcing both code paths to converge on the same
    on-disk state.
    """
    from config import (
        CORE_VARIABLES,
        ChunkingConfig,
        FillConfig,
        ProcessConfig,
        RegridConfig,
        Selection,
        StatsPeriod,
        TargetGrid,
        TimeWindow,
    )
    from process import DatasetTask, process_one
    from schema_version import SCHEMA_VERSION

    # Build synthetic source zarrs for the variables process_one needs.
    # Skip ``ta`` (we derive it from zg+hus inside the pipeline).
    day_zstores: dict[str, str] = {}
    for v in CORE_VARIABLES:
        # 3D plev variables — ua/va/hus/zg — get the plev coord.
        with_plev = v in ("ua", "va", "hus", "zg")
        # Realistic units help harmonize_temperature_to_kelvin's silent path.
        units = {
            "tas": "K",
            "huss": "1",
            "psl": "Pa",
            "pr": "kg m-2 s-1",
            "ua": "m s-1",
            "va": "m s-1",
            "hus": "1",
            "zg": "m",
        }.get(v, "1")
        day_zstores[v] = _build_source_zstore(
            tmp_path, v, units=units, with_plev=with_plev
        )

    static_zstores = {
        "sftlf": _build_static_zstore(tmp_path, "sftlf"),
        "orog": _build_static_zstore(tmp_path, "orog"),
    }

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = ProcessConfig(
        inventory_path=str(tmp_path / "inventory.csv"),
        output_directory=str(output_dir),
        # Use a tiny shared external_forcings dir so attach_external_forcings
        # silently warns and skips (no externals staged for this test).
        external_forcings_directory=str(tmp_path / "externals_unused"),
    )
    config.defaults.surface_and_ocean_variables = []  # keep the test tight
    config.defaults.time_subset = {"historical": TimeWindow("2010-01-01", "2010-12-31")}
    config.defaults.target_grid = TargetGrid(name="F22.5")
    config.defaults.regrid = RegridConfig()
    config.defaults.fill = FillConfig()
    config.defaults.chunking = ChunkingConfig(chunk_time=6, shard_time=6)
    config.defaults.max_core_missing = 0
    config.defaults.stats_periods = (StatsPeriod("full", None, None),)
    config.selection = Selection(experiments=["historical"])

    task = DatasetTask(
        source_id="TEST",
        experiment="historical",
        variant_label="r1i1p1f1",
        variant_r=1,
        variant_i=1,
        variant_p=1,
        variant_f=1,
        zstores={"day": day_zstores, "fx": static_zstores},
    )

    from index import write_sidecar

    row = process_one(task, config)
    assert row.status == "ok", f"process_one failed: {row.skip_reason}"
    assert row.schema_version == SCHEMA_VERSION
    write_sidecar(row, row.output_zarr)

    # The output zarr should already be on the current schema:
    # ``surface_temperature`` would have come from Eday.ts, which we
    # don't include here, so it won't be in the output. But ``HGTsfc``
    # (from orog) MUST be. That's the file ``process_one`` writes via
    # the CMIP_TO_OUTPUT_RENAMES path.
    fresh = xr.open_zarr(row.output_zarr, consolidated=True)
    assert "HGTsfc" in fresh.data_vars
    assert "orog" not in fresh.data_vars
    fresh_hgtsfc = fresh["HGTsfc"].values.copy()
    fresh.close()

    # Now simulate "old-schema" by inverting the rename (write the
    # same data under ``orog``), bumping the sidecar back to no
    # version, then running the migration. The migration must produce
    # the same names as the fresh pipeline output.
    sidecar_path = Path(row.output_zarr) / "metadata.json"
    aged_sidecar = json.loads(sidecar_path.read_text())
    aged_sidecar.pop("schema_version", None)
    aged_sidecar["variables_present"] = sorted(
        v if v != "HGTsfc" else "orog" for v in aged_sidecar["variables_present"]
    )

    aged = xr.open_zarr(row.output_zarr, consolidated=True).load()
    aged = aged.rename({"HGTsfc": "orog"})
    # ``to_zarr(mode="w")`` wipes the directory (including the sidecar
    # nested inside data.zarr/); persist the aged sidecar afterwards.
    aged.to_zarr(row.output_zarr, mode="w", consolidated=True, zarr_format=3)
    sidecar_path.write_text(json.dumps(aged_sidecar))

    from migrate import migrate_one

    status, detail = migrate_one(row.output_zarr)
    assert status == "migrated", f"migration failed: {detail}"

    migrated = xr.open_zarr(row.output_zarr, consolidated=True)
    assert "HGTsfc" in migrated.data_vars
    assert "orog" not in migrated.data_vars
    np.testing.assert_array_equal(migrated["HGTsfc"].values, fresh_hgtsfc)

    migrated_sidecar = json.loads(sidecar_path.read_text())
    assert migrated_sidecar["schema_version"] == SCHEMA_VERSION
    assert "HGTsfc" in migrated_sidecar["variables_present"]
    migrated.close()
