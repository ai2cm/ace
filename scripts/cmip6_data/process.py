"""Per-dataset processing for the CMIP6 daily pilot.

Reads the inventory + YAML config, selects datasets per the Selection
rules, and for each selected dataset:

1. Open core + forcing + static zstores from Pangeo's GCS mirror.
2. Validate ``cell_methods``.
3. Regrid to the Gauss-Legendre F22.5 target via xESMF.
4. Compute the time-varying below-surface mask; nearest-above fill
   the 3D plev variables.
5. Compute the derived layer-mean temperatures from ``zg`` + ``hus``.
6. Linear-interpolate the monthly forcings onto the daily axis;
   broadcast the static fields.
7. Apply the per-experiment time subset.
8. Write the zarr with zarr v3 chunk + shard sizes.
9. Drop a ``metadata.json`` sidecar as the done-marker.

Resumability
------------
Re-running the script is cheap: each target zarr is treated as
"complete" only when its ``metadata.json`` sidecar is present.
Partial zarr directories (no sidecar) are deleted and re-processed.
Pass ``--force`` to rebuild everything from scratch.

Usage
-----
    python process.py --config configs/pilot.yaml
    python process.py --config configs/pilot.yaml --dry-run
    python process.py --config configs/pilot.yaml --force
    python process.py --config configs/pilot.yaml --source-ids CanESM5 GFDL-CM4
    python process.py --config configs/pilot.yaml --max-datasets 2
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Optional

import fsspec
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import compute_and_write_stats  # noqa: E402
from config import (  # noqa: E402
    CMIP_TO_OUTPUT_RENAMES,
    SURFACE_AND_OCEAN_VARIABLES,
    ProcessConfig,
    make_label,
)
from external_forcings import attach_external_forcings  # noqa: E402
from grid import make_target_grid  # noqa: E402
from index import DatasetIndexRow, write_index, write_sidecar  # noqa: E402
from processing import (  # noqa: E402
    UNSTRUCTURED_METHOD,
    DuplicateTimestampsError,
    RssSampler,
    SimulationBoundaryError,
    apply_output_renames,
    apply_target_land_mask,
    apply_time_subset,
    clamp_static_fractions,
    compute_below_surface_mask,
    compute_derived_layer_T,
    compute_total_water_path,
    derive_ocean_and_correct_sea_ice,
    fill_derived_layer_T,
    finalize_surface_and_ocean_variable,
    flatten_plev_variables,
    grid_fingerprint,
    harmonize_temperature_to_kelvin,
    nearest_above_fill,
    normalize_plev,
    regrid_variables,
    resolve_time_duplicates,
    rss_mib,
    run_sanity_checks,
    validate_cell_methods,
    write_zarr,
)
from schema_version import SCHEMA_VERSION  # noqa: E402

# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


@dataclass
class DatasetTask:
    """One unit of work — a single (source_id, experiment, variant_label)
    slice, with all source zstore paths pre-resolved from the inventory.
    """

    source_id: str
    experiment: str
    variant_label: str
    variant_r: int
    variant_i: int
    variant_p: int
    variant_f: int

    # {table_id -> {variable_id -> zstore url}}
    zstores: dict[str, dict[str, str]] = field(default_factory=dict)

    # Native metadata from inventory (first matching row's value).
    native_grid_label: str = ""


def _load_inventory(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "zstore",
        "grid_label",
        "variant_r",
        "variant_i",
        "variant_p",
        "variant_f",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Inventory at {path} is missing columns: {sorted(missing)}")
    return df


def select_datasets(
    inventory: pd.DataFrame,
    config: ProcessConfig,
) -> list[DatasetTask]:
    """Apply Selection rules to the day-table inventory, pair forcings
    and statics, and return one :class:`DatasetTask` per surviving
    (source_id, experiment, variant_label) slice.
    """
    sel = config.selection
    day = inventory[inventory["table_id"] == "day"]
    day = day[day["experiment_id"].isin(sel.experiments)]
    if sel.source_ids is not None:
        day = day[day["source_id"].isin(sel.source_ids)]
    if sel.exclude_source_ids:
        day = day[~day["source_id"].isin(sel.exclude_source_ids)]
    if sel.exclude_variants:
        excluded = {
            (v.source_id, v.experiment, v.variant_label) for v in sel.exclude_variants
        }
        day = day[
            ~day[["source_id", "experiment_id", "member_id"]]
            .apply(tuple, axis=1)
            .isin(excluded)
        ]
    if sel.require_i is not None:
        day = day[day["variant_i"] == sel.require_i]

    # Member cap, applied per (source_id, experiment, p, f).
    if sel.max_members_per_f is not None:
        # Rank realizations inside each label slice by (f, r) ascending.
        day = day.sort_values(
            ["source_id", "experiment_id", "variant_p", "variant_f", "variant_r"]
        )
        membership = day.drop_duplicates(
            ["source_id", "experiment_id", "variant_p", "variant_f", "variant_r"]
        )[
            [
                "source_id",
                "experiment_id",
                "variant_p",
                "variant_f",
                "variant_r",
                "member_id",
            ]
        ].copy()
        membership["_rank"] = membership.groupby(
            ["source_id", "experiment_id", "variant_p", "variant_f"]
        )["variant_r"].rank(method="dense")
        kept = membership[membership["_rank"] <= sel.max_members_per_f][
            ["source_id", "experiment_id", "member_id"]
        ]
        day = day.merge(kept, on=["source_id", "experiment_id", "member_id"])

    # Build a task per (source_id, experiment, member_id).
    tasks: list[DatasetTask] = []
    dataset_keys = day.drop_duplicates(["source_id", "experiment_id", "member_id"])

    for _, row in dataset_keys.iterrows():
        source_id = row["source_id"]
        experiment = row["experiment_id"]
        variant_label = row["member_id"]
        day_slice = day[
            (day["source_id"] == source_id)
            & (day["experiment_id"] == experiment)
            & (day["member_id"] == variant_label)
        ]
        zstores: dict[str, dict[str, str]] = {"day": {}}
        for _, r in day_slice.iterrows():
            zstores["day"][r["variable_id"]] = r["zstore"]
        # Fold any CFday-sourced day-cadence variables (rsdt/rsut) into
        # the same ``day`` zstore dict so downstream code can look them
        # up by name without knowing the underlying table. Prefer day
        # over CFday on the rare chance both publish; in practice no
        # model publishes rsdt/rsut on the standard day table.
        cfday_slice = inventory[
            (inventory["table_id"] == "CFday")
            & (inventory["source_id"] == source_id)
            & (inventory["experiment_id"] == experiment)
            & (inventory["member_id"] == variant_label)
        ]
        for _, r in cfday_slice.iterrows():
            zstores["day"].setdefault(r["variable_id"], r["zstore"])

        # Surface-and-ocean source tables (Amon, Eday, SImon, SIday, Oday,
        # Omon) — require matching variant_label when possible; fall
        # back to any matching member otherwise; else drop.
        surface_and_ocean_tables = sorted(
            {h.table_id for h in SURFACE_AND_OCEAN_VARIABLES}
        )
        for table in surface_and_ocean_tables:
            zstores[table] = {}
            cands = inventory[
                (inventory["table_id"] == table)
                & (inventory["source_id"] == source_id)
                & (inventory["experiment_id"] == experiment)
            ]
            exact = cands[cands["member_id"] == variant_label]
            picked = exact if len(exact) else cands
            # Prefer r1 fallback within the matching (p, f).
            if len(picked):
                picked = picked.sort_values(["variant_p", "variant_f", "variant_r"])
                for _, r in picked.drop_duplicates("variable_id").iterrows():
                    zstores[table][r["variable_id"]] = r["zstore"]

        # Statics (fx) — variant-agnostic per-model.
        zstores["fx"] = {}
        stat = inventory[
            (inventory["table_id"] == "fx") & (inventory["source_id"] == source_id)
        ]
        if len(stat):
            stat = stat.sort_values("variant_p")
            for _, r in stat.drop_duplicates("variable_id").iterrows():
                zstores["fx"][r["variable_id"]] = r["zstore"]

        tasks.append(
            DatasetTask(
                source_id=source_id,
                experiment=experiment,
                variant_label=variant_label,
                variant_r=int(row["variant_r"]),
                variant_i=int(row["variant_i"]),
                variant_p=int(row["variant_p"]),
                variant_f=int(row["variant_f"]),
                zstores=zstores,
                native_grid_label=str(row["grid_label"]),
            )
        )

    tasks.sort(key=lambda t: (t.source_id, t.experiment, t.variant_label))
    return tasks


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def output_zarr_path(output_directory: str, task: DatasetTask) -> str:
    return (
        f"{output_directory.rstrip('/')}/"
        f"{task.source_id}/{task.experiment}/{task.variant_label}/data.zarr"
    )


def _sidecar_path(zarr_path: str) -> str:
    return f"{zarr_path.rstrip('/')}/metadata.json"


def _fs_exists(path: str) -> bool:
    fs, rel = fsspec.core.url_to_fs(path)
    return fs.exists(rel)


def _fs_delete_tree(path: str) -> None:
    fs, rel = fsspec.core.url_to_fs(path)
    if fs.exists(rel):
        fs.rm(rel, recursive=True)


def _load_existing_sidecar(zarr_path: str) -> Optional[DatasetIndexRow]:
    sidecar = _sidecar_path(zarr_path)
    if not _fs_exists(sidecar):
        return None
    with fsspec.open(sidecar, "r") as f:
        data = json.load(f)
    # DatasetIndexRow fields are a superset of what we serialized; load
    # what we recognise and leave the rest at defaults.
    allowed = {f.name for f in dataclasses.fields(DatasetIndexRow)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return DatasetIndexRow(**filtered)


def _scan_all_sidecars(output_directory: str) -> list[DatasetIndexRow]:
    """Walk ``output_directory`` for every ``metadata.json`` sidecar
    written by prior runs and return the deserialised rows. Used so
    the central index always reflects the full on-disk state, even
    when the current run only touched a subset of datasets.
    """
    fs, rel = fsspec.core.url_to_fs(output_directory.rstrip("/"))
    if not fs.exists(rel):
        return []
    pattern = f"{rel}/**/metadata.json"
    paths = fs.glob(pattern)
    rows: list[DatasetIndexRow] = []
    allowed = {f.name for f in dataclasses.fields(DatasetIndexRow)}
    required = {
        f.name
        for f in dataclasses.fields(DatasetIndexRow)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    for p in paths:
        with fs.open(p, "r") as f:
            data = json.load(f)
        if not required.issubset(data):
            # Non-dataset sidecar (e.g. external_forcings/<exp>/metadata.json);
            # skip rather than fail.
            continue
        filtered = {k: v for k, v in data.items() if k in allowed}
        rows.append(DatasetIndexRow(**filtered))
    return rows


def _merge_rows_for_index(
    run_rows: list[DatasetIndexRow],
    output_directory: str,
) -> list[DatasetIndexRow]:
    """Combine this run's rows with every sidecar on disk.

    Precedence: for a given ``(source_id, experiment, variant_label)``
    triple, the current run's row wins (it may reflect a fresh
    failure / skip even when an older sidecar exists). Sidecars for
    datasets not touched this run are included verbatim so the index
    stays complete when running with ``--source-ids``.
    """

    def _key(r: DatasetIndexRow) -> tuple[str, str, str]:
        return (r.source_id, r.experiment, r.variant_label)

    this_run = {_key(r): r for r in run_rows}
    combined: dict[tuple[str, str, str], DatasetIndexRow] = {}
    for row in _scan_all_sidecars(output_directory):
        combined[_key(row)] = row
    combined.update(this_run)
    return sorted(combined.values(), key=_key)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _open_zstore(url: str) -> xr.Dataset:
    """Open a Pangeo GCS zarr with cftime-decoded times.

    We force ``use_cftime=True`` because CMIP6 ssp585 extends to 2100
    and the time axis ends up as ``cftime.datetime`` for some models
    (``dates out of range`` for numpy's ``datetime64[ns]``, which
    maxes out at year 2262 but xarray's default decoder picks up
    extrapolated bounds that overrun it for a few publishers). Mixing
    cftime day-table times with datetime64 Amon times later blows up
    ``interp(time=...)`` with a ``UFuncTypeError``. All-cftime avoids
    that.
    """
    mapper = fsspec.get_mapper(url)
    # ``chunks={"time": 365}`` forces dask-backed lazy arrays — without
    # explicit chunks, xesmf's regridder can materialize the entire
    # variable into RAM during the call, OOM'ing models with large
    # native grids (AWI-ESM, etc.).
    return xr.open_zarr(
        mapper, consolidated=True, use_cftime=True, chunks={"time": 365}
    )


def process_one(task: DatasetTask, config: ProcessConfig) -> DatasetIndexRow:
    """Full pipeline for one dataset. Exceptions are caught and reflected
    as ``status=failed`` rows so one bad dataset doesn't block the rest.
    """
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    label = make_label(task.source_id, task.variant_p)
    zarr_path = output_zarr_path(config.output_directory, task)

    # Build surface_and_ocean_zstores: output_name -> source zstore URL for every
    # surface-and-ocean variable whose source table is published for this
    # task. Variables whose source isn't published are silently absent.
    surface_and_ocean_zstores: dict[str, str] = {}
    for h in SURFACE_AND_OCEAN_VARIABLES:
        if h.output_name not in cfg.surface_and_ocean_variables:
            continue
        zs = task.zstores.get(h.table_id, {}).get(h.var_id)
        if zs:
            surface_and_ocean_zstores[h.output_name] = zs

    row = DatasetIndexRow(
        source_id=task.source_id,
        experiment=task.experiment,
        variant_label=task.variant_label,
        variant_r=task.variant_r,
        variant_i=task.variant_i,
        variant_p=task.variant_p,
        variant_f=task.variant_f,
        label=label,
        target_grid=cfg.target_grid.name,
        native_grid_label=task.native_grid_label,
        core_zstores=[
            task.zstores.get("day", {}).get(v, "") for v in cfg.core_variables
        ],
        surface_and_ocean_zstores=surface_and_ocean_zstores,
        static_zstores=[
            task.zstores.get("fx", {}).get(v, "") for v in cfg.static_variables
        ],
        output_zarr=zarr_path,
        status="pending",
    )

    process_t0 = time.monotonic()

    def _stage(label: str, t0: float) -> None:
        """Record a per-stage wall-clock split. Logs the delta and the
        cumulative since ``process_t0``, plus current RSS, so the
        longest / heaviest stages are obvious without grep-summing the
        log."""
        now = time.monotonic()
        logging.info(
            "  [stage %s] +%.1fs (cum %.1fs, rss %.0f MiB)",
            label,
            now - t0,
            now - process_t0,
            rss_mib(),
        )

    sampler = RssSampler()
    sampler.start()
    try:
        # 1. Gate on required core variables. Missing variables are
        # tolerated up to ``cfg.max_core_missing``; the remaining ones
        # are simply absent from the output (training handles via
        # per-sample variable masking).
        day_zs = task.zstores.get("day", {})
        missing_core = [v for v in cfg.core_variables if v not in day_zs]
        if len(missing_core) > cfg.max_core_missing:
            row.status = "skipped"
            row.skip_reason = (
                f"missing {len(missing_core)} > {cfg.max_core_missing} core "
                f"variables: {missing_core}"
            )
            return row
        if missing_core:
            row.warnings.append(f"core variables absent (tolerated): {missing_core}")

        # 2. Open + group-by-grid regrid. Naively merging before
        #    regridding breaks on models that publish different
        #    day-table variables on different native grids (e.g.
        #    HadGEM3-GC31-LL has ua/va on staggered u-points and
        #    scalars on cell centres — ``join="inner"`` of their
        #    lats is empty). Per-variable regrid is safe but builds
        #    one xesmf ``Regridder`` per variable per method, ~30
        #    ESMF calls per dataset and dramatically slow.
        stage_t0 = time.monotonic()
        #
        #    Compromise: bucket variables by their native grid
        #    fingerprint and regrid each bucket as one sub-dataset.
        #    Most models have one or two distinct grids across their
        #    day variables, so this stays at 2-4 Regridder builds per
        #    dataset (one per (grid, method)) while still handling
        #    staggered grids correctly.
        all_day_vars = [v for v in cfg.core_variables if v in day_zs] + [
            v for v in cfg.optional_variables if v in day_zs
        ]
        target = make_target_grid(cfg.target_grid.name)
        row.cell_methods_mismatch = []
        groups: dict[Hashable, list[xr.Dataset]] = {}
        try:
            for v in all_day_vars:
                ds_v = _open_zstore(day_zs[v])[[v]]

                # cell_methods check (source-side; post-regrid values
                # are conceptually still "time: mean").
                mismatches = validate_cell_methods(ds_v, [v])
                row.cell_methods_mismatch.extend(mismatches)

                # Capture native calendar from the first opened variable.
                if not row.native_calendar and "time" in ds_v.dims:
                    try:
                        row.native_calendar = str(ds_v["time"].dt.calendar)
                    except (AttributeError, TypeError) as e:
                        row.warnings.append(f"could not read native calendar: {e}")

                # Normalise plev order (no-op on 2D vars).
                ds_v = normalize_plev(ds_v)

                # Time-subset BEFORE dedupe so the dedupe loop only
                # checks duplicates within the (small) window we're
                # going to use. CESM2-WACCM has hundreds of duplicate
                # timestamps spread across its 165-year historical
                # store, and loading every duplicate's full 3D field
                # from GCS to compare values dominates the run time.
                ds_v = apply_time_subset(ds_v, cfg)
                if ds_v.sizes.get("time", 0) == 0:
                    row.status = "skipped"
                    row.skip_reason = f"no timesteps in configured time_subset for {v}"
                    return row

                # Time dedupe (raises SimulationBoundaryError for
                # non-identical duplicates; caught below).
                ds_v, msg = resolve_time_duplicates(
                    ds_v, v, allow_dedupe=cfg.allow_dedupe
                )
                if msg:
                    row.warnings.append(msg)

                groups.setdefault(grid_fingerprint(ds_v), []).append(ds_v)
        except (SimulationBoundaryError, DuplicateTimestampsError) as e:
            row.status = "skipped"
            row.skip_reason = str(e)
            return row

        if row.cell_methods_mismatch:
            row.warnings.append(
                f"cell_methods != 'time: mean' for {row.cell_methods_mismatch}"
            )
        if len(groups) > 1:
            row.warnings.append(
                f"day vars span {len(groups)} distinct native grids "
                "(staggered or otherwise); regridded per-group to F22.5"
            )

        # Per-grid merge (within a group, all vars share lat/lon, so
        # ``join="inner"`` is a spatial no-op) and regrid.
        regridded_pieces: list[xr.Dataset] = []
        for grid_dss in groups.values():
            merged_grid = xr.merge(grid_dss, compat="override", join="inner")
            piece, methods = regrid_variables(merged_grid, target, cfg)
            row.regrid_methods.update(methods)
            regridded_pieces.append(piece)

        # Final merge across grid groups — all on F22.5 now, so the
        # spatial join is trivial; only the time axis is intersected.
        day_regridded = xr.merge(regridded_pieces, compat="override", join="inner")

        if "time" in day_regridded.dims and day_regridded.sizes["time"] == 0:
            row.status = "skipped"
            row.skip_reason = (
                "empty merged time dim — source variables have "
                "non-overlapping time ranges (publisher quirk)"
            )
            return row

        _stage("regrid_core (lazy graph build)", stage_t0)

        # 8. Static fields from fx — regrid once, drop time.
        static_ds: Optional[xr.Dataset] = None
        fx_zs = task.zstores.get("fx", {})
        fx_have = [v for v in cfg.static_variables if v in fx_zs]
        if fx_have:
            fx_pieces = [
                _open_zstore(fx_zs[v])[[v]].squeeze(drop=True) for v in fx_have
            ]
            fx_merged = xr.merge(fx_pieces, compat="override")
            static_regridded, static_methods = regrid_variables(fx_merged, target, cfg)
            static_regridded, clip_warnings = clamp_static_fractions(static_regridded)
            row.warnings.extend(clip_warnings)
            static_ds = static_regridded
            row.regrid_methods.update(static_methods)

        # All the steps from here through write_zarr trigger dask
        # compute eventually (via ``.compute()`` on reductions or
        # ``.to_zarr()`` streaming). xesmf's Regridder wraps the ESMF C
        # library and is *not* thread-safe — concurrent regridder calls
        # produce NaN chunks silently. We previously dodged this by
        # full-``.load()``-ing the dataset up front, but that doesn't
        # fit at prod scale (75–86 years × all variables ≫ 32 GB).
        # The synchronous dask scheduler forces serial chunk evaluation,
        # which is safe for xesmf and streams memory through one chunk
        # at a time instead of materialising everything.
        #
        # Apply it here (rather than later, just before write_zarr) so
        # that compute_below_surface_mask's ``.any()`` reduction — the
        # first all-data compute after regrid — also runs serially. On
        # native-resolution high-res sources (EC-Earth3 T255,
        # GFDL-CM4 C192) the default threaded scheduler eagerly loads
        # too many regridded 3D chunks in parallel and OOMs the pod
        # even though the result is a single boolean.
        import dask

        dask.config.set(scheduler="synchronous")

        # 9. Below-surface mask computed from the un-filled 3D fields.
        stage_t0 = time.monotonic()
        orog: Optional[xr.DataArray] = None
        if static_ds is not None and "orog" in static_ds:
            orog = static_ds["orog"]
        mask, row.mask_source = compute_below_surface_mask(day_regridded, orog)
        _stage("below_surface_mask", stage_t0)

        # 10. Derive layer-mean T from the un-filled zg + hus. Running
        # this on filled zg would force dz = 0 in below-surface columns
        # (nearest-above fill copies zg[i+1] down, so zg[i] == zg[i+1])
        # and collapse the derived T to zero. Do it first, then fill.
        have_derived_T = "zg" in day_regridded and "hus" in day_regridded
        if have_derived_T:
            day_regridded = compute_derived_layer_T(day_regridded)

        # 11. Nearest-above fill for the level-valued 3D state.
        # When no mask is available (no orog, no NaN pattern), skip
        # filling and don't write a mask variable.
        if mask is not None:
            for v in ("ua", "va", "hus", "zg"):
                if v in day_regridded:
                    day_regridded[v] = nearest_above_fill(day_regridded[v], mask)
            day_regridded = day_regridded.assign(below_surface_mask=mask)

            # 12. Cascading fill for the derived layer T — layer i is
            # invalid where either bounding level is below surface.
            if have_derived_T:
                day_regridded = fill_derived_layer_T(day_regridded, mask)

        # 13. Surface-and-ocean variables (surface T, sea-ice, ocean). Each
        # variable is opened from its source table, regridded, mapped to
        # the daily axis according to its cadence (drop-in for ``daily``,
        # causal previous-month-mean for ``monthly_causal``), and
        # filled+masked for ocean/sea-ice kinds. Variables whose source
        # table is not published for this dataset are silently absent.
        stage_t0 = time.monotonic()
        #
        # Catch per-variable so one failure (e.g., a tripolar grid that
        # trips xesmf with ``ESMC_FieldRegridStore failed``) doesn't
        # block the rest of the dataset.
        daily_time = day_regridded["time"]
        for h in SURFACE_AND_OCEAN_VARIABLES:
            if h.output_name not in cfg.surface_and_ocean_variables:
                continue
            z = task.zstores.get(h.table_id, {}).get(h.var_id)
            if not z:
                # This dataset's model simply doesn't publish this
                # source. No warning — the variable is just absent.
                continue
            try:
                # Keep the full dataset (not ``[[var]]``) so cell-bounds
                # coords like ``lat_bnds`` / ``vertices_longitude`` —
                # which carry an extra ``d2`` / ``vertices`` dim the
                # variable itself doesn't have — survive the open and
                # are available to xesmf's conservative regridder.
                src = _open_zstore(z)
                src, dup_msg = resolve_time_duplicates(
                    src, h.var_id, allow_dedupe=cfg.allow_dedupe
                )
                if dup_msg:
                    row.warnings.append(dup_msg)
                src = apply_time_subset(src, cfg)
                drop = [v for v in src.data_vars if v != h.var_id]
                src = src.drop_vars(drop, errors="ignore")
                regridded, methods_used = regrid_variables(src, target, cfg)
                # Record regrid method under the output name so
                # downstream tooling can match without knowing the
                # source-table rename.
                for sv, meth in methods_used.items():
                    if sv == h.var_id:
                        row.regrid_methods[h.output_name] = meth
                    else:
                        row.regrid_methods[sv] = meth
                regridded_var = regridded[h.var_id]
                # Unstructured ocean sources (FESOM) have no land cells,
                # so xesmf's locstream nearest fills every target cell
                # with the nearest ocean value. Restore the NaN-over-land
                # pattern from the target-grid land_fraction so
                # emit_mask_and_fill produces a correct mask. Without
                # land_fraction we skip masking and warn — the variable
                # is still valid over ocean.
                if methods_used.get(h.var_id) == UNSTRUCTURED_METHOD and h.kind in (
                    "ocean_surface",
                    "seaice_surface",
                ):
                    if static_ds is not None and "land_fraction" in static_ds:
                        regridded_var = apply_target_land_mask(
                            regridded_var, static_ds["land_fraction"]
                        )
                    else:
                        row.warnings.append(
                            f"{h.output_name}: unstructured source regridded "
                            "via nearest_s2d but no target land_fraction "
                            "available; mask channel will be all-ones"
                        )
                outputs = finalize_surface_and_ocean_variable(
                    regridded_var,
                    h,
                    daily_time,
                    fill_iterations=cfg.fill.ocean_fill_iterations,
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "  surface-and-ocean %s (%s.%s) failed: %s",
                    h.output_name,
                    h.table_id,
                    h.var_id,
                    e,
                )
                row.warnings.append(
                    f"{h.output_name} from {h.table_id}.{h.var_id} failed: "
                    f"{type(e).__name__}: {e}"
                )
                continue
            for name, da in outputs.items():
                day_regridded[name] = da
        _stage("surface_and_ocean_loop", stage_t0)

        # 14. Attach static fields (broadcast along time implicitly at read).
        # After clamp_static_fractions, the static set holds ``land_fraction``
        # (rescaled to [0, 1]) instead of CMIP's ``sftlf``.
        if static_ds is not None:
            for v in static_ds.data_vars:
                day_regridded[v] = static_ds[v]

        # 14_pre. Derived ``total_water_path = water_vapor_path +
        # clwvi`` when both are present — emits the CM4/SHIELD-style
        # total-water field while keeping the CMIP6 split intact.
        if "water_vapor_path" in day_regridded and "clwvi" in day_regridded:
            day_regridded["total_water_path"] = compute_total_water_path(
                day_regridded["water_vapor_path"], day_regridded["clwvi"]
            )

        # 14a. Derive ``{simon,siday}_ocean_fraction`` from
        # ``land_fraction`` and the corresponding sea-ice fraction so
        # the (land, ocean, sea-ice) triple is in the output. Coastal
        # cells where land+ice exceeds 1 (typically from the
        # emit_mask_and_fill diffusion over land) get their excess
        # moved back into sea_ice_fraction so the identity
        # land+ice+ocean=1 holds exactly.
        if "land_fraction" in day_regridded:
            for sif, ofv in (
                ("simon_sea_ice_fraction", "simon_ocean_fraction"),
                ("siday_sea_ice_fraction", "siday_ocean_fraction"),
            ):
                if sif in day_regridded:
                    corrected_ice, ocean = derive_ocean_and_correct_sea_ice(
                        day_regridded["land_fraction"],
                        day_regridded[sif],
                        ofv,
                    )
                    day_regridded[sif] = corrected_ice
                    day_regridded[ofv] = ocean

        # 14b. External forcings (input4MIPs / LUH2). Currently CO2 only.
        # The per-scenario zarr is staged once by ``external_forcings.py``
        # at ``<output_directory>/external_forcings/<experiment>.zarr``;
        # if it isn't present we skip silently with a warning, so a run
        # without staged externals still produces datasets (just missing
        # the input4mips_* variables).
        attach_external_forcings(
            day_regridded,
            row,
            config.resolved_external_forcings_directory,
            task.experiment,
        )

        # 15. Sanity count of NaN cells in the filled 3D state.
        nan_total = 0
        for v in ("ua", "va", "hus", "zg"):
            if v in day_regridded:
                nan_total += int(day_regridded[v].isnull().sum().compute())
        row.n_nan_input_cells = nan_total

        # 16. Populate time metadata.
        row.n_timesteps = int(day_regridded.sizes.get("time", 0))
        if row.n_timesteps:
            row.time_start = str(day_regridded["time"].values[0])
            row.time_end = str(day_regridded["time"].values[-1])

        # 17. Flatten plev dimensions into pressure-named 2D variables.
        day_regridded = flatten_plev_variables(day_regridded)

        # 18. Harmonize all temperature variables to K. CMIP6 spec
        # publishes ``tos``/``tob``/``sitemptop`` in °C and the
        # rest (``tas``, ``ts``, ...) in K, but publishers
        # occasionally deviate; walking the assembled dataset
        # catches the variants via their ``units`` attribute.
        # Idempotent on already-K vars. Skip ``_mask`` channels —
        # they're 0/1 indicators that inherited their parent's
        # ``units`` from ``emit_mask_and_fill`` (since fixed at
        # source) and have no business being temperature-shifted.
        for v in list(day_regridded.data_vars):
            if v.endswith("_mask"):
                continue
            da, msg = harmonize_temperature_to_kelvin(day_regridded[v], var_id=v)
            if msg:
                row.warnings.append(msg)
                if "converted" in msg:
                    day_regridded[v] = da

        # 19. Rename CMIP6 variables to the upstream-baseline
        # convention so downstream training can share variable
        # names with SHIELD/ERA5 datasets.
        day_regridded = apply_output_renames(day_regridded, CMIP_TO_OUTPUT_RENAMES)

        # 20. Sanity checks — advisory only. Run *after* renames
        # and K-harmonization so ``_SANITY_RANGES`` keys off the
        # final variable names and unit conventions.
        sanity = run_sanity_checks(day_regridded)
        if sanity:
            row.warnings.extend(sanity)
            for msg in sanity:
                logging.warning("  sanity: %s", msg)

        # 21. Stream-write the dataset to zarr. With the synchronous
        # dask scheduler set above, each chunk is evaluated and
        # persisted in turn — peak memory stays at a single chunk × N
        # variables instead of the full dataset.
        stage_t0 = time.monotonic()
        day_regridded.attrs["label"] = label
        day_regridded.attrs["source_id"] = task.source_id
        day_regridded.attrs["experiment"] = task.experiment
        day_regridded.attrs["variant_label"] = task.variant_label
        logging.info("  streaming zarr write...")
        write_zarr(day_regridded, zarr_path, cfg)
        row.variables_present = sorted(day_regridded.data_vars)
        _stage("write_zarr", stage_t0)

        # 22. Compute per-dataset stats inline. We *re-open* the
        # just-written zarr rather than reusing the in-memory dataset
        # — the on-disk version is at target resolution (small) with
        # no upstream dask graph attached, so the stats pass scans it
        # cheaply without re-triggering any regrid work.
        stage_t0 = time.monotonic()
        stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
        try:
            written = xr.open_zarr(zarr_path, consolidated=True)
            compute_and_write_stats(
                written,
                stats_path,
                identity={
                    "source_id": task.source_id,
                    "experiment": task.experiment,
                    "variant_label": task.variant_label,
                    "label": label,
                },
                grid_name=cfg.target_grid.name,
                periods=tuple(cfg.stats_periods),
            )
            written.close()
        except Exception as e:  # noqa: BLE001
            row.warnings.append(f"inline stats failed: {type(e).__name__}: {e}")
            logging.warning("  inline stats failed for %s: %s", zarr_path, e)
        _stage("inline_stats", stage_t0)

        row.schema_version = SCHEMA_VERSION
        row.status = "ok"
        logging.info(
            "Finished %s in %.1fs total",
            zarr_path,
            time.monotonic() - process_t0,
        )
        return row

    except Exception as e:  # noqa: BLE001
        logging.exception("Processing failed for %s", zarr_path)
        row.status = "failed"
        row.skip_reason = f"{type(e).__name__}: {e}"
        row.warnings.append("".join(traceback.format_exception_only(type(e), e)))
        return row
    finally:
        sampler.stop()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    config: ProcessConfig,
    tasks: list[DatasetTask],
    *,
    force: bool = False,
) -> list[DatasetIndexRow]:
    rows: list[DatasetIndexRow] = []
    for i, task in enumerate(tasks, start=1):
        zarr_path = output_zarr_path(config.output_directory, task)
        logging.info(
            "[%d/%d] %s %s %s",
            i,
            len(tasks),
            task.source_id,
            task.experiment,
            task.variant_label,
        )

        if not force:
            existing = _load_existing_sidecar(zarr_path)
            if existing is not None:
                logging.info("  already complete, skipping (use --force to rebuild)")
                rows.append(existing)
                continue

        if _fs_exists(zarr_path):
            logging.info("  partial zarr without sidecar — deleting and retrying")
            _fs_delete_tree(zarr_path)

        row = process_one(task, config)
        rows.append(row)
        if row.status == "ok":
            write_sidecar(row, zarr_path)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to process YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected tasks and exit without processing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even completed datasets (ignores metadata.json sidecars).",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=None,
        help="Optional subset of source_ids to process (overrides config).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional subset of experiments to process (overrides config).",
    )
    parser.add_argument(
        "--variant-labels",
        nargs="+",
        default=None,
        help="Optional subset of variant labels to process (post-selection filter).",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit to this many datasets (after selection). Debug aid.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip writing the central index. Useful when running many "
        "parallel jobs that each process a single dataset; a separate "
        "rebuild_index.py run consolidates the sidecars afterward.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Bump xarray's file-handle LRU well past the per-task worst case
    # so we don't race the cache during compute. Mirrors process_esgf.py.
    xr.set_options(file_cache_maxsize=1024)

    config = ProcessConfig.from_file(args.config)
    if args.source_ids is not None:
        config.selection.source_ids = args.source_ids
    if args.experiments is not None:
        config.selection.experiments = args.experiments

    inventory = _load_inventory(config.inventory_path)
    tasks = select_datasets(inventory, config)
    if args.variant_labels is not None:
        allowed = set(args.variant_labels)
        tasks = [t for t in tasks if t.variant_label in allowed]
    if args.max_datasets is not None:
        tasks = tasks[: args.max_datasets]

    logging.info("Selected %d tasks.", len(tasks))

    if args.dry_run:
        for t in tasks:
            print(f"{t.source_id}\t{t.experiment}\t{t.variant_label}")
        return

    rows = run(config, tasks, force=args.force)

    if not args.skip_index:
        # Central index: merge this-run's rows with every sidecar on disk so
        # narrow --source-ids / --max-datasets re-runs don't truncate the
        # index to the subset just processed.
        all_rows = _merge_rows_for_index(rows, config.output_directory)
        write_index(all_rows, config.output_directory)

    # Final status summary covers only datasets attempted in THIS run.
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    if args.skip_index:
        logging.info("Done. This run: %s (index update skipped).", by_status)
    else:
        logging.info(
            "Done. This run: %s; index now lists %d datasets.",
            by_status,
            len(all_rows),
        )


if __name__ == "__main__":
    main()
