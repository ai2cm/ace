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
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fsspec
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import compute_and_write_stats  # noqa: E402
from config import (  # noqa: E402
    CMIP_TO_OUTPUT_RENAMES,
    SURFACE_AND_OCEAN_VARIABLES,
    ESGFProcessConfig,
    cmip6_source_table,
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
from index import (  # noqa: E402
    DatasetIndexRow,
    clear_failure_record,
    write_failure_record,
    write_index,
    write_sidecar,
)
from processing import (  # noqa: E402
    BOUNDS_NAMES,
    UNSTRUCTURED_METHOD,
    DuplicateTimestampsError,
    RssSampler,
    SimulationBoundaryError,
    apply_output_renames,
    apply_target_land_mask,
    apply_time_subset,
    clamp_static_fractions,
    compute_below_surface_mask,
    compute_total_water_path,
    derive_ocean_and_correct_sea_ice,
    finalize_surface_and_ocean_variable,
    flatten_plev_variables,
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
        # CFday-sourced day-cadence variables (rsdt/rsut). Fold the
        # variables published on CFday for this (model, experiment,
        # member) into ``available_vars`` so the all_day_vars loop
        # picks them up; ``cmip6_source_table`` resolves the table
        # at download time.
        cfday_slice = inventory[
            (inventory["table_id"] == "CFday")
            & (inventory["source_id"] == source_id)
            & (inventory["experiment_id"] == experiment)
            & (inventory["member_id"] == member)
        ]
        available_vars |= set(cfday_slice["variable_id"].unique())
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
        # process_one_esgf only runs when no Pangeo zarr is present; the
        # whole dataset is sourced from ESGF.
        data_source="esgf",
    )

    scratch = scratch_dir_for_dataset(
        config.esgf.scratch_dir,
        task.source_id,
        task.experiment,
        task.variant_label,
    )

    process_t0 = time.monotonic()

    def _stage(label: str, t0: float) -> None:
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
        target = make_target_grid(cfg.target_grid.name)

        # 1. Process all daily variables one at a time. Missing core
        # variables (up to ``cfg.max_core_missing``) are tolerated and
        # tracked as warnings.
        stage_t0 = time.monotonic()
        all_day_vars = [
            v for v in cfg.core_variables if v in task.available_day_variables
        ] + [v for v in cfg.optional_variables if v in task.available_day_variables]

        regridded_vars: dict[str, xr.Dataset] = {}
        for v in all_day_vars:
            try:
                source_table = cmip6_source_table(v)
                logging.info(
                    "  [%s] processing %s/%s ...", task.source_id, source_table, v
                )
                result, methods = _download_and_regrid_variable(
                    task, v, source_table, target, config, scratch
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

        _stage("download_and_regrid_day_vars", stage_t0)

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

        # All the steps from here through write_zarr trigger dask
        # compute eventually. See process.py for the full rationale —
        # short version: xesmf is not thread-safe, the full ``.load()``
        # we used to do up front doesn't fit at prod scale, so the
        # synchronous dask scheduler streams chunks one at a time
        # through the regrid + write path.
        #
        # Apply it here (rather than just before write_zarr) so that
        # compute_below_surface_mask's ``.any()`` reduction — the first
        # all-data compute — also runs serially.
        import dask

        dask.config.set(scheduler="synchronous")

        # 5. Below-surface mask.
        stage_t0 = time.monotonic()
        orog: Optional[xr.DataArray] = None
        if static_ds is not None and "orog" in static_ds:
            orog = static_ds["orog"]
        mask, row.mask_source = compute_below_surface_mask(day_regridded, orog)
        _stage("below_surface_mask", stage_t0)

        # 6. Nearest-above fill for the level-valued 3D state.
        if mask is not None:
            for v in ("ua", "va", "hus", "zg"):
                if v in day_regridded:
                    day_regridded[v] = nearest_above_fill(day_regridded[v], mask)
            day_regridded = day_regridded.assign(below_surface_mask=mask)

        # 7. Surface-and-ocean variables (surface T, sea-ice, ocean) — see
        # the matching block in process.py for the design. ESGF picks
        # the source by table/var pair and downloads via
        # ``_download_and_regrid_variable``; the post-regrid logic is
        # shared with the Pangeo pipeline via
        # ``finalize_surface_and_ocean_variable``.
        stage_t0 = time.monotonic()
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
        _stage("surface_and_ocean_loop", stage_t0)

        # 9. Attach static fields.
        if static_ds is not None:
            for v in static_ds.data_vars:
                day_regridded[v] = static_ds[v]

        # 9_pre. Derived total_water_path = water_vapor_path + clwvi
        # when both are present (CM4/SHIELD-style total-water field).
        if "water_vapor_path" in day_regridded and "clwvi" in day_regridded:
            day_regridded["total_water_path"] = compute_total_water_path(
                day_regridded["water_vapor_path"], day_regridded["clwvi"]
            )

        # 9a. Derive {simon,siday}_ocean_fraction + budget-correct
        # sea-ice fraction so land+ice+ocean=1 (see matching block in
        # process.py).
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

        # 9b. External forcings (input4MIPs / LUH2). See the matching
        # block in process.py — staging is done once globally by
        # ``external_forcings.py`` and the per-scenario zarr is
        # opportunistically attached here.
        attach_external_forcings(
            day_regridded,
            row,
            config.resolved_external_forcings_directory,
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

        # 12. Flatten plev.
        day_regridded = flatten_plev_variables(day_regridded)

        # 13. Harmonize temperatures to K (some CMIP6 publishers
        # emit ``tos``/``tob``/``sitemptop`` in °C). See process.py.
        # Skip ``_mask`` channels — they're 0/1 indicators that
        # used to inherit ``units`` from their parent variable.
        for v in list(day_regridded.data_vars):
            if v.endswith("_mask"):
                continue
            da, msg = harmonize_temperature_to_kelvin(day_regridded[v], var_id=v)
            if msg:
                row.warnings.append(msg)
                if "converted" in msg:
                    day_regridded[v] = da

        # 14. Rename CMIP6 variables to the baseline convention
        # (see process.py for rationale).
        day_regridded = apply_output_renames(day_regridded, CMIP_TO_OUTPUT_RENAMES)

        # 15. Sanity checks — advisory only. Run *after* renames
        # and K-harmonization.
        sanity = run_sanity_checks(day_regridded)
        if sanity:
            row.warnings.extend(sanity)
            for msg in sanity:
                logging.warning("  sanity: %s", msg)

        # 16. Stream-write the dataset to zarr.
        stage_t0 = time.monotonic()
        day_regridded.attrs["label"] = label
        day_regridded.attrs["source_id"] = task.source_id
        day_regridded.attrs["experiment"] = task.experiment
        day_regridded.attrs["variant_label"] = task.variant_label
        day_regridded.attrs["data_source"] = "esgf"
        logging.info("  streaming zarr write...")
        write_zarr(day_regridded, zarr_path, cfg)
        row.variables_present = sorted(day_regridded.data_vars)
        _stage("write_zarr", stage_t0)

        # 17. Inline per-dataset stats; re-open the just-written zarr
        # so the stats pass scans target-resolution data on disk
        # rather than triggering any upstream regrid work.
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
        # Full-ESGF dataset: every variable came from the ESGF pipeline.
        # Mirror this in the audit list so downstream consumers don't
        # have to special-case ``data_source == "esgf"``.
        row.esgf_augmented_variables = sorted(day_regridded.data_vars)
        logging.info(
            "Finished %s in %.1fs total",
            zarr_path,
            time.monotonic() - process_t0,
        )

    except Exception as e:  # noqa: BLE001
        logging.exception("Processing failed for %s", zarr_path)
        row.status = "failed"
        row.skip_reason = f"{type(e).__name__}: {e}"
        row.warnings.append("".join(traceback.format_exception_only(type(e), e)))
    finally:
        sampler.stop()
        cleanup_scratch_dir(scratch)

    return row


def augment_one_esgf(
    task: ESGFDatasetTask,
    config: ESGFProcessConfig,
    existing_row: DatasetIndexRow,
) -> DatasetIndexRow:
    """Augment an existing Pangeo zarr with ESGF surface-and-ocean
    variables it didn't have.

    Strategy:

    - Open the existing zarr to get the daily time axis + target grid.
    - For each ``SurfaceAndOceanVariable`` whose ``output_name`` is in
      ``task.available_surface_and_ocean_variables`` but not in
      ``existing_row.variables_present``, download + regrid + finalize
      from ESGF and ``to_zarr(mode="a")`` it onto the existing zarr.
    - Stamp ``data_source=pangeo+esgf``, ``esgf_augmented_variables``
      in the sidecar and as zarr root attrs so downstream consumers
      can tell what came from where.
    - Regenerate ``stats.nc`` so the new variables are covered.

    Limitations (v1):
    - Only surface-and-ocean variables. 3D plev day variables (ua, va,
      hus, zg) are skipped because they require running the flatten
      step against the existing zarr's plev convention — left for a
      follow-up.
    - Statics (HGTsfc, land_fraction) are not augmented; the ESGF
      pipeline would need to re-derive ocean fractions which is too
      entangled with the day-data path to bolt on cleanly here.

    Caller is responsible for ensuring the existing zarr's
    ``schema_version`` matches the current ``SCHEMA_VERSION`` —
    augmenting an older-schema zarr is blocked at the call site so
    we don't mix on-disk conventions.
    """
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    zarr_path = _output_zarr_path(config.output_directory, task)

    existing_vars = set(existing_row.variables_present)
    augmentable: list = []
    for h in SURFACE_AND_OCEAN_VARIABLES:
        if h.output_name not in cfg.surface_and_ocean_variables:
            continue
        if h.output_name not in task.available_surface_and_ocean_variables:
            continue
        if h.output_name in existing_vars:
            continue
        augmentable.append(h)

    if not augmentable:
        logging.info("  no ESGF augmentation available; leaving Pangeo zarr as-is")
        return existing_row

    logging.info(
        "  augmenting with %d ESGF surface-and-ocean variables: %s",
        len(augmentable),
        ", ".join(h.output_name for h in augmentable),
    )

    scratch = scratch_dir_for_dataset(
        config.esgf.scratch_dir,
        task.source_id,
        task.experiment,
        task.variant_label,
    )

    target = make_target_grid(cfg.target_grid.name)
    existing_ds = xr.open_zarr(zarr_path, consolidated=True)
    daily_time = existing_ds["time"]

    added_names: list[str] = []
    sampler = RssSampler()
    sampler.start()
    try:
        import dask

        dask.config.set(scheduler="synchronous")
        for h in augmentable:
            try:
                logging.info(
                    "  [%s] augment %s/%s -> %s ...",
                    task.source_id,
                    h.table_id,
                    h.var_id,
                    h.output_name,
                )
                result, methods = _download_and_regrid_variable(
                    task, h.var_id, h.table_id, target, config, scratch
                )
                existing_row.regrid_methods.update(
                    {
                        h.output_name if k == h.var_id else k: v
                        for k, v in methods.items()
                    }
                )
                if result is None:
                    continue
                regridded_var = result[h.var_id]
                outputs = finalize_surface_and_ocean_variable(
                    regridded_var,
                    h,
                    daily_time,
                    fill_iterations=cfg.fill.ocean_fill_iterations,
                )
                # to_zarr(mode='a') one variable at a time keeps memory
                # bounded and gives a clean checkpoint per variable in
                # case the augment is interrupted partway through.
                new_ds = xr.Dataset({name: da for name, da in outputs.items()})
                new_ds.to_zarr(
                    zarr_path,
                    mode="a",
                    consolidated=False,
                    zarr_format=3,
                    align_chunks=True,
                )
                added_names.extend(outputs.keys())
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "  augment %s failed: %s: %s — skipping this variable",
                    h.output_name,
                    type(e).__name__,
                    e,
                )
                existing_row.warnings.append(
                    f"augment {h.output_name} failed: {type(e).__name__}: {e}"
                )
        existing_ds.close()
    finally:
        sampler.stop()
        cleanup_scratch_dir(scratch)

    if not added_names:
        logging.info("  no variables successfully added — leaving zarr unchanged")
        return existing_row

    # Consolidate metadata once after all variable appends.
    import zarr

    zarr.consolidate_metadata(zarr_path)

    # Stamp dataset-level attrs so the augmentation is visible from
    # the zarr itself (not just the sidecar). Open the just-written
    # zarr to apply attrs at the group root.
    group = zarr.open_group(zarr_path, mode="r+")
    prior_aug = list(existing_row.esgf_augmented_variables)
    all_aug = sorted(set(prior_aug) | set(added_names))
    group.attrs["data_source"] = "pangeo+esgf"
    group.attrs["esgf_augmented_variables"] = all_aug
    zarr.consolidate_metadata(zarr_path)

    # Refresh stats.nc so the new variables get per-dataset stats.
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
                "label": existing_row.label,
            },
            grid_name=cfg.target_grid.name,
            periods=tuple(cfg.stats_periods),
        )
        written.close()
    except Exception as e:  # noqa: BLE001
        existing_row.warnings.append(
            f"stats regeneration after augment failed: {type(e).__name__}: {e}"
        )
        logging.warning("  stats regeneration failed after augment: %s", e)

    existing_row.data_source = "pangeo+esgf"
    existing_row.esgf_augmented_variables = all_aug
    existing_row.variables_present = sorted(existing_vars | set(added_names))
    return existing_row


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
                # Existing zarr from the Pangeo pipeline (or a prior
                # ESGF run): try to augment with ESGF variables Pangeo
                # didn't have. Gate on schema parity — mixing on-disk
                # conventions across schema versions would corrupt the
                # archive, so require migrate.py to bring the dataset
                # to SCHEMA_VERSION first.
                if existing.schema_version != SCHEMA_VERSION:
                    logging.info(
                        "  schema_version=%r != current %r — augment "
                        "requires migrate.py first; skipping",
                        existing.schema_version,
                        SCHEMA_VERSION,
                    )
                    rows.append(existing)
                    continue
                row = augment_one_esgf(task, config, existing)
                rows.append(row)
                # Augment never produces failures here; if augment
                # failed for individual variables, those are recorded
                # in row.warnings and the row stays status=ok.
                write_sidecar(row, zarr_path)
                clear_failure_record(row, config.output_directory)
                continue

        if _fs_exists(zarr_path):
            logging.info("  partial zarr without sidecar — deleting and retrying")
            _fs_delete_tree(zarr_path)

        row = process_one_esgf(task, config)
        rows.append(row)
        if row.status == "ok":
            write_sidecar(row, zarr_path)
            clear_failure_record(row, config.output_directory)
        else:
            write_failure_record(row, config.output_directory)

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

    # The per-variable lazy-open path accumulates a NetCDF file handle
    # per file across the all_day_vars loop — for CMCC-ESM2-like models
    # with ~30 vars × ~10 historical chunks each, that's ~300 handles
    # held until the final ``write_zarr`` triggers compute. xarray's
    # default LRU cache size is 128; once exceeded, cache evictions
    # mid-read can race with the dask compute and raise ``KeyError``
    # in ``file_manager._acquire_with_cache_info`` (the obscure
    # ``KeyError: [<class 'netCDF4._netCDF4.Dataset'>, (...,), ...]``).
    # Bumping the cap well past the worst-case file count avoids it.
    xr.set_options(file_cache_maxsize=1024)

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
