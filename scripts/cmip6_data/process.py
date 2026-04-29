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
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Optional

import fsspec
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import ProcessConfig, make_label  # noqa: E402
from grid import make_target_grid  # noqa: E402
from index import DatasetIndexRow, write_index, write_sidecar  # noqa: E402
from processing import (  # noqa: E402
    DuplicateTimestampsError,
    SimulationBoundaryError,
    apply_time_subset,
    compute_below_surface_mask,
    compute_derived_layer_T,
    fill_derived_layer_T,
    flatten_plev_variables,
    grid_fingerprint,
    interp_monthly_to_daily,
    nearest_above_fill,
    normalize_plev,
    regrid_variables,
    resolve_time_duplicates,
    run_sanity_checks,
    validate_cell_methods,
    write_zarr,
)

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

        # Forcings (Amon, SImon) — require matching variant_label when
        # possible; fall back to r1i1p<p>f<f> otherwise; else drop.
        for table in ("Amon", "SImon"):
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
    for p in paths:
        with fs.open(p, "r") as f:
            data = json.load(f)
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
    return xr.open_zarr(mapper, consolidated=True, use_cftime=True)


def process_one(task: DatasetTask, config: ProcessConfig) -> DatasetIndexRow:
    """Full pipeline for one dataset. Exceptions are caught and reflected
    as ``status=failed`` rows so one bad dataset doesn't block the rest.
    """
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    label = make_label(task.source_id, task.variant_p)
    zarr_path = output_zarr_path(config.output_directory, task)

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
        forcing_interpolation=cfg.forcing_interpolation,
        core_zstores=[
            task.zstores.get("day", {}).get(v, "") for v in cfg.core_variables
        ],
        forcing_zstores=[
            task.zstores.get("Amon", {}).get("ts", ""),
            task.zstores.get("SImon", {}).get("siconc", ""),
        ],
        static_zstores=[
            task.zstores.get("fx", {}).get(v, "") for v in cfg.static_variables
        ],
        output_zarr=zarr_path,
        status="pending",
    )

    try:
        # 1. Gate on required core variables.
        day_zs = task.zstores.get("day", {})
        missing_core = [v for v in cfg.core_variables if v not in day_zs]
        if missing_core:
            row.status = "skipped"
            row.skip_reason = f"missing core variables: {missing_core}"
            return row

        # 2. Open + group-by-grid regrid. Naively merging before
        #    regridding breaks on models that publish different
        #    day-table variables on different native grids (e.g.
        #    HadGEM3-GC31-LL has ua/va on staggered u-points and
        #    scalars on cell centres — ``join="inner"`` of their
        #    lats is empty). Per-variable regrid is safe but builds
        #    one xesmf ``Regridder`` per variable per method, ~30
        #    ESMF calls per dataset and dramatically slow.
        #
        #    Compromise: bucket variables by their native grid
        #    fingerprint and regrid each bucket as one sub-dataset.
        #    Most models have one or two distinct grids across their
        #    day variables, so this stays at 2-4 Regridder builds per
        #    dataset (one per (grid, method)) while still handling
        #    staggered grids correctly.
        all_day_vars = cfg.core_variables + [
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
            static_ds = static_regridded
            row.regrid_methods.update(static_methods)

        # 9. Below-surface mask computed from the un-filled 3D fields.
        orog: Optional[xr.DataArray] = None
        if static_ds is not None and "orog" in static_ds:
            orog = static_ds["orog"]
        mask, row.mask_source = compute_below_surface_mask(day_regridded, orog)

        # 10. Derive layer-mean T from the un-filled zg + hus. Running
        # this on filled zg would force dz = 0 in below-surface columns
        # (nearest-above fill copies zg[i+1] down, so zg[i] == zg[i+1])
        # and collapse the derived T to zero. Do it first, then fill.
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
            day_regridded = fill_derived_layer_T(day_regridded, mask)

        # 13. Monthly forcings -> daily. Curvilinear ocean grids (e.g.
        # SImon siconc on a tripolar grid) occasionally trip xesmf's
        # regridder with ``ESMC_FieldRegridStore failed`` or similar.
        # Catch per-forcing so the rest of the dataset still gets
        # written; the model just loses that one forcing variable.
        for table, var in (("Amon", "ts"), ("SImon", "siconc")):
            if var not in cfg.forcing_variables:
                continue
            z = task.zstores.get(table, {}).get(var)
            if not z:
                row.warnings.append(f"missing forcing {var} from {table}")
                continue
            try:
                # Keep the full dataset (not ``[[var]]``) so cell-bounds
                # coords like ``lat_bnds`` / ``vertices_longitude`` —
                # which carry an extra ``d2`` / ``vertices`` dim the
                # variable itself doesn't have — survive the open and
                # are available to xesmf's conservative regridder.
                monthly = _open_zstore(z)
                monthly, dup_msg = resolve_time_duplicates(
                    monthly, var, allow_dedupe=cfg.allow_dedupe
                )
                if dup_msg:
                    row.warnings.append(dup_msg)
                monthly = apply_time_subset(monthly, cfg)
                # Keep only ``var`` as a data_var; everything else can
                # live as coords or be dropped by the regridder.
                drop = [v for v in monthly.data_vars if v != var]
                monthly = monthly.drop_vars(drop, errors="ignore")
                monthly_r, methods_used = regrid_variables(monthly, target, cfg)
                row.regrid_methods.update(methods_used)
                interp = interp_monthly_to_daily(
                    monthly_r[var], day_regridded["time"], cfg.forcing_interpolation
                )
            except Exception as e:  # noqa: BLE001
                logging.warning("  forcing %s from %s failed: %s", var, table, e)
                row.warnings.append(
                    f"forcing {var} from {table} failed: {type(e).__name__}: {e}"
                )
                continue
            day_regridded[var] = interp

        # 13b. Sea-ice fraction is only defined over ocean cells — emit
        # a time-invariant ``siconc_mask`` from the regridded ocean-grid
        # coverage, then replace the residual land NaN with 0 so the
        # stored data is NaN-free. The mask is a 2D (lat, lon) uint8
        # (1 = valid / ocean, 0 = land or missing source coverage); we
        # take it from the first timestep, since the valid-cell pattern
        # is time-invariant for sea-ice fraction. We deliberately do
        # *not* clip the ocean values to [0, 100] — conservative
        # regridding's output preserves the area-weighted integral and
        # clipping would break that. Rounding can leave sub-epsilon
        # values outside the nominal range; the sanity checks tolerate
        # that.
        if "siconc" in day_regridded:
            valid = (~day_regridded["siconc"].isel(time=0).isnull()).astype("uint8")
            day_regridded = day_regridded.assign(
                siconc_mask=valid.rename("siconc_mask")
            )
            day_regridded["siconc"] = day_regridded["siconc"].fillna(0.0)

        # 14. Attach static fields (broadcast along time implicitly at read).
        if static_ds is not None:
            for v in static_ds.data_vars:
                day_regridded[v] = static_ds[v]

        # 15. Sanity count of NaN cells in the filled 3D state. Should
        # be zero; any non-zero is a sign the fill logic missed
        # something. Uses eager .compute() because dask arrays don't
        # support .item().
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

        # 17. Materialize the whole dataset before writing. xesmf's
        # Regridder wraps the ESMF C library, which is not thread-safe;
        # leaving the regrid lazy in the dask graph lets dask parallelise
        # per-chunk, and the resulting concurrent Regridder calls can
        # silently produce NaN chunks. Loading up front runs the regrid
        # once, sequentially, and gives deterministic output. It also
        # lets the sanity checks run on materialised data cheaply.
        logging.info("  materializing dataset in memory before write...")
        day_regridded = day_regridded.load()

        # 18. Sanity checks — advisory only. Failures go to warnings.
        sanity = run_sanity_checks(day_regridded)
        if sanity:
            row.warnings.extend(sanity)
            for msg in sanity:
                logging.warning("  sanity: %s", msg)

        # 19. Flatten plev dimensions into pressure-named 2D variables.
        day_regridded = flatten_plev_variables(day_regridded)

        # 20. Write zarr + record variables.
        day_regridded.attrs["label"] = label
        day_regridded.attrs["source_id"] = task.source_id
        day_regridded.attrs["experiment"] = task.experiment
        day_regridded.attrs["variant_label"] = task.variant_label
        write_zarr(day_regridded, zarr_path, cfg)
        row.variables_present = sorted(day_regridded.data_vars)
        row.status = "ok"
        return row

    except Exception as e:  # noqa: BLE001
        logging.exception("Processing failed for %s", zarr_path)
        row.status = "failed"
        row.skip_reason = f"{type(e).__name__}: {e}"
        row.warnings.append("".join(traceback.format_exception_only(type(e), e)))
        return row


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
