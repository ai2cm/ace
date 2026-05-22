"""ESGF-sourced processing for the CMIP6 daily pipeline.

Processes CMIP6 datasets sourced from ESGF rather than Pangeo's GCS
zarr mirror. Uses a variable-at-a-time strategy to keep disk usage
under ~50 GB:

1. Download static fields (orog, sftlf) — tiny, kept throughout.
2. For each daily variable, for each time-chunk file:
   a. Download the NetCDF from ESGF.
   b. Regrid to the target Gauss-Legendre grid (reusing cached weights).
   c. Delete the native file.
3. Assemble the regridded variables into a single dataset.
4. Compute the below-surface mask and nearest-above fill.
5. Read back regridded zg to compute derived layer-mean T.
6. Process monthly forcings (ts, siconc).
7. Run sanity checks, flatten plev, and write the output zarr.

Resumability: re-running skips datasets whose ``metadata.json`` sidecar
already exists (same as ``process.py``). The ESGF pipeline writes into
the same output directory structure so both pipelines share a unified
index.

Usage:
    python process_esgf.py --config configs/process_esgf.yaml
    python process_esgf.py --config configs/process_esgf.yaml --dry-run
    python process_esgf.py --config configs/process_esgf.yaml --source-ids CanESM5-1
"""

import argparse
import dataclasses
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fsspec
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    SURFACE_AND_OCEAN_VARIABLES,
    ESGFProcessConfig,
    make_label,
)
from esgf import (  # noqa: E402
    cleanup_scratch_dir,
    cleanup_variable_files,
    download_file,
    filter_files_by_time,
    query_files,
    scratch_dir_for_dataset,
)
from external_forcings import attach_external_forcings  # noqa: E402
from grid import make_target_grid  # noqa: E402
from index import DatasetIndexRow, write_index, write_sidecar  # noqa: E402
from processing import (  # noqa: E402
    BOUNDS_NAMES,
    UNSTRUCTURED_METHOD,
    DuplicateTimestampsError,
    SimulationBoundaryError,
    apply_target_land_mask,
    apply_time_subset,
    clamp_static_fractions,
    compute_below_surface_mask,
    compute_derived_layer_T,
    fill_derived_layer_T,
    finalize_surface_and_ocean_variable,
    flatten_plev_variables,
    nearest_above_fill,
    normalize_plev,
    regrid_variables,
    resolve_time_duplicates,
    run_sanity_checks,
    validate_cell_methods,
    write_zarr,
)

# ---------------------------------------------------------------------------
# Task building from ESGF inventory
# ---------------------------------------------------------------------------


@dataclass
class ESGFDatasetTask:
    """One unit of work for the ESGF pipeline."""

    source_id: str
    experiment: str
    variant_label: str
    variant_r: int
    variant_i: int
    variant_p: int
    variant_f: int
    grid_label: str = ""
    available_day_variables: list[str] = field(default_factory=list)
    # Output names of surface-and-ocean variables this dataset's source
    # tables actually publish (e.g. ``["amon_ts", "eday_ts",
    # "simon_siconc"]``).
    available_surface_and_ocean_variables: list[str] = field(default_factory=list)
    has_orog: bool = False
    has_sftlf: bool = False


def _load_esgf_inventory(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
        "variant_r",
        "variant_i",
        "variant_p",
        "variant_f",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Inventory at {path} missing columns: {sorted(missing)}")
    return df


def select_esgf_datasets(
    inventory: pd.DataFrame,
    config: ESGFProcessConfig,
) -> list[ESGFDatasetTask]:
    """Apply selection rules and build tasks from an ESGF inventory."""
    sel = config.selection
    day = inventory[inventory["table_id"] == "day"]
    day = day[day["experiment_id"].isin(sel.experiments)]
    if sel.source_ids is not None:
        day = day[day["source_id"].isin(sel.source_ids)]
    if sel.exclude_source_ids:
        day = day[~day["source_id"].isin(sel.exclude_source_ids)]
    if sel.require_i is not None:
        day = day[day["variant_i"] == sel.require_i]

    if sel.max_members_per_f is not None:
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

    # Core variables gate. Missing variables are tolerated up to
    # ``config.defaults.max_core_missing``; remaining ones are absent
    # from output (training handles via per-sample masking).
    core = set(config.defaults.core_variables)
    max_missing = config.defaults.max_core_missing

    tasks: list[ESGFDatasetTask] = []
    keys = day.drop_duplicates(["source_id", "experiment_id", "member_id"])

    for _, row in keys.iterrows():
        source_id = row["source_id"]
        experiment = row["experiment_id"]
        member = row["member_id"]

        day_slice = day[
            (day["source_id"] == source_id)
            & (day["experiment_id"] == experiment)
            & (day["member_id"] == member)
        ]
        available_vars = set(day_slice["variable_id"].unique())
        if len(core - available_vars) > max_missing:
            continue

        grid_labels = day_slice["grid_label"].unique()
        grid_label = grid_labels[0] if len(grid_labels) else ""

        # Check surface-and-ocean source tables and statics.
        model_inv = inventory[inventory["source_id"] == source_id]
        exp_inv = model_inv[model_inv["experiment_id"] == experiment]
        available_surface_and_ocean: list[str] = []
        for h in SURFACE_AND_OCEAN_VARIABLES:
            published = bool(
                len(
                    exp_inv[
                        (exp_inv["table_id"] == h.table_id)
                        & (exp_inv["variable_id"] == h.var_id)
                    ]
                )
            )
            if published:
                available_surface_and_ocean.append(h.output_name)
        has_orog = bool(
            len(
                model_inv[
                    (model_inv["table_id"] == "fx")
                    & (model_inv["variable_id"] == "orog")
                ]
            )
        )
        has_sftlf = bool(
            len(
                model_inv[
                    (model_inv["table_id"] == "fx")
                    & (model_inv["variable_id"] == "sftlf")
                ]
            )
        )

        tasks.append(
            ESGFDatasetTask(
                source_id=source_id,
                experiment=experiment,
                variant_label=member,
                variant_r=int(row["variant_r"]),
                variant_i=int(row["variant_i"]),
                variant_p=int(row["variant_p"]),
                variant_f=int(row["variant_f"]),
                grid_label=grid_label,
                available_day_variables=sorted(available_vars),
                available_surface_and_ocean_variables=available_surface_and_ocean,
                has_orog=has_orog,
                has_sftlf=has_sftlf,
            )
        )

    tasks.sort(key=lambda t: (t.source_id, t.experiment, t.variant_label))
    return tasks


# ---------------------------------------------------------------------------
# Resumability (shared with process.py)
# ---------------------------------------------------------------------------


def _output_zarr_path(output_directory: str, task: ESGFDatasetTask) -> str:
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
    allowed = {f.name for f in dataclasses.fields(DatasetIndexRow)}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return DatasetIndexRow(**filtered)


def _scan_all_sidecars(output_directory: str) -> list[DatasetIndexRow]:
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
    def _key(r: DatasetIndexRow) -> tuple[str, str, str]:
        return (r.source_id, r.experiment, r.variant_label)

    this_run = {_key(r): r for r in run_rows}
    combined: dict[tuple[str, str, str], DatasetIndexRow] = {}
    for row in _scan_all_sidecars(output_directory):
        combined[_key(row)] = row
    combined.update(this_run)
    return sorted(combined.values(), key=_key)


# ---------------------------------------------------------------------------
# Variable-at-a-time processing
# ---------------------------------------------------------------------------


def _open_netcdf_files(paths: list[Path], variable: str) -> xr.Dataset:
    """Open and concatenate a list of NetCDF files along time.

    ``chunks={"time": 365}`` forces dask-backed lazy arrays; without
    explicit chunks ``xr.open_dataset`` returns numpy-backed lazy
    arrays whose fancy-index reads (``apply_time_subset`` uses
    ``isel(time=np.where(mask)[0])``) materialize the entire variable
    in RAM, OOM-ing models with large files (IPSL 15.5 GB ua, etc.).
    """
    datasets = []
    for p in sorted(paths):
        ds = xr.open_dataset(
            p,
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            chunks={"time": 365},
        )
        if variable in ds.data_vars:
            keep = [variable] + [v for v in ds.data_vars if v in BOUNDS_NAMES]
            datasets.append(ds[keep])
    if not datasets:
        raise ValueError(f"No data found for {variable} in {len(paths)} files")
    return xr.concat(datasets, dim="time", data_vars="minimal")


def _download_and_regrid_variable(
    task: ESGFDatasetTask,
    variable: str,
    table_id: str,
    target_grid: xr.Dataset,
    config: ESGFProcessConfig,
    scratch: Path,
) -> tuple[Optional[xr.Dataset], dict[str, str]]:
    """Download, regrid, and return one variable. Cleans up native files.

    Returns ``(regridded_ds, {variable: method})``."""
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    node = config.esgf.search_node

    logging.info("    querying ESGF for %s/%s ...", table_id, variable)
    fileset = query_files(
        node, task.source_id, task.experiment, task.variant_label, table_id, variable
    )
    if not fileset.files:
        return None, {}

    tw = cfg.time_subset.get(task.experiment)
    if tw is not None:
        n_before = len(fileset.files)
        fileset = filter_files_by_time(fileset, tw.start, tw.end)
        if n_before != len(fileset.files):
            logging.info(
                "    time filter: %d → %d files (of %d)",
                n_before,
                len(fileset.files),
                n_before,
            )
        if not fileset.files:
            return None, {}

    logging.info(
        "    downloading %d files (%.1f GB) for %s ...",
        len(fileset.files),
        fileset.total_size / 1e9,
        variable,
    )

    local_paths: list[Path] = []
    for f in fileset.files:
        local_paths.append(download_file(f, scratch))

    logging.info("    opening and concatenating %s ...", variable)
    ds = _open_netcdf_files(local_paths, variable)
    ds = normalize_plev(ds)
    if table_id == "fx":
        if "time" in ds.dims:
            ds = ds.isel(time=0, drop=True)
    else:
        ds = apply_time_subset(ds, cfg)
        if ds.sizes.get("time", 0) == 0:
            cleanup_variable_files(scratch, variable)
            return None, {}
        ds, msg = resolve_time_duplicates(ds, variable, allow_dedupe=cfg.allow_dedupe)
        if msg:
            logging.warning("    %s", msg)

    logging.info("    regridding %s ...", variable)
    regridded, methods = regrid_variables(ds, target_grid, cfg)
    regridded = regridded.load()

    cleanup_variable_files(scratch, variable)
    return regridded, methods


def process_one_esgf(
    task: ESGFDatasetTask,
    config: ESGFProcessConfig,
) -> DatasetIndexRow:
    """Full pipeline for one ESGF dataset."""
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    label = make_label(task.source_id, task.variant_p)
    zarr_path = _output_zarr_path(config.output_directory, task)

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
        native_grid_label=task.grid_label,
        # surface_and_ocean_zstores is left empty for ESGF — provenance is captured
        # implicitly by ``available_surface_and_ocean_variables`` and the file-level
        # ESGF queries resolved at download time.
        output_zarr=zarr_path,
        status="pending",
    )

    scratch = scratch_dir_for_dataset(
        config.esgf.scratch_dir,
        task.source_id,
        task.experiment,
        task.variant_label,
    )

    try:
        target = make_target_grid(cfg.target_grid.name)

        # 1. Process all daily variables one at a time. Missing core
        # variables (up to ``cfg.max_core_missing``) are tolerated and
        # tracked as warnings.
        all_day_vars = [
            v for v in cfg.core_variables if v in task.available_day_variables
        ] + [v for v in cfg.optional_variables if v in task.available_day_variables]

        regridded_vars: dict[str, xr.Dataset] = {}
        for v in all_day_vars:
            try:
                logging.info("  [%s] processing day/%s ...", task.source_id, v)
                result, methods = _download_and_regrid_variable(
                    task, v, "day", target, config, scratch
                )
                row.regrid_methods.update(methods)
                if result is not None:
                    regridded_vars[v] = result
            except (SimulationBoundaryError, DuplicateTimestampsError) as e:
                row.status = "skipped"
                row.skip_reason = str(e)
                cleanup_scratch_dir(scratch)
                return row

        missing_core = [v for v in cfg.core_variables if v not in regridded_vars]
        if len(missing_core) > cfg.max_core_missing:
            row.status = "skipped"
            row.skip_reason = (
                f"missing {len(missing_core)} > {cfg.max_core_missing} core "
                f"variables after processing: {missing_core}"
            )
            cleanup_scratch_dir(scratch)
            return row
        if missing_core:
            row.warnings.append(f"core variables absent (tolerated): {missing_core}")

        # 2. Merge all regridded daily variables.
        day_regridded = xr.merge(
            list(regridded_vars.values()), compat="override", join="inner"
        )

        if "time" in day_regridded.dims and day_regridded.sizes["time"] == 0:
            row.status = "skipped"
            row.skip_reason = "empty time after merge"
            cleanup_scratch_dir(scratch)
            return row

        # Capture native calendar.
        if "time" in day_regridded.dims:
            try:
                row.native_calendar = str(day_regridded["time"].dt.calendar)
            except (AttributeError, TypeError):
                pass

        # 3. Cell methods validation.
        mm = validate_cell_methods(day_regridded, all_day_vars)
        if mm:
            row.cell_methods_mismatch = mm
            row.warnings.append(f"cell_methods != 'time: mean' for {mm}")

        # 4. Static fields.
        static_ds: Optional[xr.Dataset] = None
        for static_var in cfg.static_variables:
            has_it = (static_var == "orog" and task.has_orog) or (
                static_var == "sftlf" and task.has_sftlf
            )
            if not has_it:
                continue
            logging.info("  [%s] processing fx/%s ...", task.source_id, static_var)
            result, methods = _download_and_regrid_variable(
                task, static_var, "fx", target, config, scratch
            )
            row.regrid_methods.update(methods)
            if result is not None:
                if static_ds is None:
                    static_ds = result
                else:
                    static_ds = xr.merge([static_ds, result], compat="override")
        if static_ds is not None:
            static_ds, clip_warnings = clamp_static_fractions(static_ds)
            row.warnings.extend(clip_warnings)

        # 5. Below-surface mask.
        orog: Optional[xr.DataArray] = None
        if static_ds is not None and "orog" in static_ds:
            orog = static_ds["orog"]
        mask, row.mask_source = compute_below_surface_mask(day_regridded, orog)

        # 6. Derived layer-mean T from un-filled zg + hus (only when
        # both are present).
        have_derived_T = "zg" in day_regridded and "hus" in day_regridded
        if have_derived_T:
            day_regridded = compute_derived_layer_T(day_regridded)

        # 7. Nearest-above fill.
        if mask is not None:
            for v in ("ua", "va", "hus", "zg"):
                if v in day_regridded:
                    day_regridded[v] = nearest_above_fill(day_regridded[v], mask)
            day_regridded = day_regridded.assign(below_surface_mask=mask)
            if have_derived_T:
                day_regridded = fill_derived_layer_T(day_regridded, mask)

        # 8. Surface-and-ocean variables (surface T, sea-ice, ocean) — see
        # the matching block in process.py for the design. ESGF picks
        # the source by table/var pair and downloads via
        # ``_download_and_regrid_variable``; the post-regrid logic is
        # shared with the Pangeo pipeline via
        # ``finalize_surface_and_ocean_variable``.
        daily_time = day_regridded["time"]
        for h in SURFACE_AND_OCEAN_VARIABLES:
            if h.output_name not in cfg.surface_and_ocean_variables:
                continue
            if h.output_name not in task.available_surface_and_ocean_variables:
                continue
            try:
                logging.info(
                    "  [%s] processing %s/%s -> %s ...",
                    task.source_id,
                    h.table_id,
                    h.var_id,
                    h.output_name,
                )
                result, methods = _download_and_regrid_variable(
                    task, h.var_id, h.table_id, target, config, scratch
                )
                # Map regrid method to output name.
                for sv, meth in methods.items():
                    if sv == h.var_id:
                        row.regrid_methods[h.output_name] = meth
                    else:
                        row.regrid_methods[sv] = meth
                if result is None:
                    continue
                regridded_var = result[h.var_id]
                if methods.get(h.var_id) == UNSTRUCTURED_METHOD and h.kind in (
                    "ocean_surface",
                    "seaice_surface",
                ):
                    if static_ds is not None and "sftlf" in static_ds:
                        regridded_var = apply_target_land_mask(
                            regridded_var, static_ds["sftlf"]
                        )
                    else:
                        row.warnings.append(
                            f"{h.output_name}: unstructured source regridded "
                            "via nearest_s2d but no target sftlf available; "
                            "mask channel will be all-ones"
                        )
                outputs = finalize_surface_and_ocean_variable(
                    regridded_var,
                    h,
                    daily_time,
                    fill_iterations=cfg.fill.ocean_fill_iterations,
                )
                for name, da in outputs.items():
                    day_regridded[name] = da
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

        # 9. Attach static fields.
        if static_ds is not None:
            for v in static_ds.data_vars:
                day_regridded[v] = static_ds[v]

        # 9b. External forcings (input4MIPs / LUH2). See the matching
        # block in process.py — staging is done once globally by
        # ``external_forcings.py`` and the per-scenario zarr is
        # opportunistically attached here.
        attach_external_forcings(
            day_regridded,
            row,
            config.output_directory,
            task.experiment,
        )

        # 10. NaN count.
        nan_total = 0
        for v in ("ua", "va", "hus", "zg"):
            if v in day_regridded:
                nan_total += int(day_regridded[v].isnull().sum().compute())
        row.n_nan_input_cells = nan_total

        # 11. Time metadata.
        row.n_timesteps = int(day_regridded.sizes.get("time", 0))
        if row.n_timesteps:
            row.time_start = str(day_regridded["time"].values[0])
            row.time_end = str(day_regridded["time"].values[-1])

        # 12. Materialize.
        logging.info("  materializing dataset in memory before write...")
        day_regridded = day_regridded.load()

        # 13. Sanity checks.
        sanity = run_sanity_checks(day_regridded)
        if sanity:
            row.warnings.extend(sanity)
            for msg in sanity:
                logging.warning("  sanity: %s", msg)

        # 14. Flatten plev.
        day_regridded = flatten_plev_variables(day_regridded)

        # 15. Write zarr.
        day_regridded.attrs["label"] = label
        day_regridded.attrs["source_id"] = task.source_id
        day_regridded.attrs["experiment"] = task.experiment
        day_regridded.attrs["variant_label"] = task.variant_label
        day_regridded.attrs["data_source"] = "esgf"
        write_zarr(day_regridded, zarr_path, cfg)
        row.variables_present = sorted(day_regridded.data_vars)
        row.status = "ok"

    except Exception as e:  # noqa: BLE001
        logging.exception("Processing failed for %s", zarr_path)
        row.status = "failed"
        row.skip_reason = f"{type(e).__name__}: {e}"
        row.warnings.append("".join(traceback.format_exception_only(type(e), e)))
    finally:
        cleanup_scratch_dir(scratch)

    return row


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    config: ESGFProcessConfig,
    tasks: list[ESGFDatasetTask],
    *,
    force: bool = False,
) -> list[DatasetIndexRow]:
    rows: list[DatasetIndexRow] = []
    for i, task in enumerate(tasks, start=1):
        zarr_path = _output_zarr_path(config.output_directory, task)
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

        row = process_one_esgf(task, config)
        rows.append(row)
        if row.status == "ok":
            write_sidecar(row, zarr_path)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", required=True, help="Path to ESGF inventory CSV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-ids", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--variant-labels", nargs="+", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = ESGFProcessConfig.from_file(args.config)
    if args.source_ids is not None:
        config.selection.source_ids = args.source_ids
    if args.experiments is not None:
        config.selection.experiments = args.experiments

    inventory = _load_esgf_inventory(args.inventory)
    tasks = select_esgf_datasets(inventory, config)
    if args.variant_labels is not None:
        allowed = set(args.variant_labels)
        tasks = [t for t in tasks if t.variant_label in allowed]
    if args.max_datasets is not None:
        tasks = tasks[: args.max_datasets]

    logging.info("Selected %d ESGF tasks.", len(tasks))

    if args.dry_run:
        for t in tasks:
            day_vars = ",".join(t.available_day_variables)
            so = ",".join(t.available_surface_and_ocean_variables)
            print(
                f"{t.source_id}\t{t.experiment}\t{t.variant_label}\t"
                f"orog={t.has_orog}\tsftlf={t.has_sftlf}\t"
                f"day=[{day_vars}]\tsurface_and_ocean=[{so}]"
            )
        return

    rows = run(config, tasks, force=args.force)

    if not args.skip_index:
        all_rows = _merge_rows_for_index(rows, config.output_directory)
        write_index(all_rows, config.output_directory)

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
