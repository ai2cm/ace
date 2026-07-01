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
from concurrent.futures import ThreadPoolExecutor
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
    MASKED_3D_STATE_FIELDS,
    UNSTRUCTURED_METHOD,
    DuplicateTimestampsError,
    RssSampler,
    SimulationBoundaryError,
    align_to_reference_grid,
    apply_output_renames,
    apply_target_land_mask,
    apply_time_subset,
    assemble_masked_3d_state,
    attach_mask_attributes,
    clamp_static_fractions,
    compute_below_surface_mask,
    compute_total_water_path,
    decode_default_fills,
    derive_ocean_and_correct_sea_ice,
    fill_below_surface_smooth,
    finalize_surface_and_ocean_variable,
    flatten_plev_variables,
    harmonize_temperature_to_kelvin,
    make_regridder,
    normalize_plev,
    regrid_variables,
    resolve_time_duplicates,
    rss_mib,
    run_sanity_checks,
    source_grid_masked_regrid,
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
            ds = ds[keep]
            # Defend against publisher fill-value leaks; same rationale
            # as ``_open_zstore`` in process.py. CESM2 ``omon_tob``
            # files ship raw 9.97e+36 fills without a ``_FillValue``
            # attribute set, so xarray's mask_and_scale never decodes
            # them.
            ds, fill_warnings = decode_default_fills(ds)
            for msg in fill_warnings:
                logging.warning("  %s: %s", p.name, msg)
            datasets.append(ds)
    if not datasets:
        raise ValueError(f"No data found for {variable} in {len(paths)} files")
    return xr.concat(datasets, dim="time", data_vars="minimal")


def _download_files(files, scratch: Path, n_workers: int) -> list[Path]:
    """Download ESGF files to ``scratch``, returning local paths in input order.

    ``download_file`` writes each file (and its ``.partial``) to a distinct
    path, so concurrent calls do not collide. ``executor.map`` preserves input
    order — keeping the returned paths time-ordered for concatenation — and
    re-raises the first download failure exactly as a serial loop would.
    """
    if n_workers <= 1 or len(files) <= 1:
        return [download_file(f, scratch) for f in files]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(lambda f: download_file(f, scratch), files))


def _download_native_variable(
    task: ESGFDatasetTask,
    variable: str,
    table_id: str,
    config: ESGFProcessConfig,
    scratch: Path,
) -> Optional[xr.Dataset]:
    """Download + open one variable on its NATIVE grid (no regrid, no cleanup).

    Returns the native dataset (plev-normalised, time-subset, dedup'd) or
    ``None`` when ESGF has no files or the time filter empties the set. The
    caller owns regridding and native-file cleanup (via
    ``cleanup_variable_files``) — the source-grid masking path needs several
    native fields resident at once, so cleanup can't happen per variable here.
    """
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    node = config.esgf.search_node

    logging.info("    querying ESGF for %s/%s ...", table_id, variable)
    fileset = query_files(
        node, task.source_id, task.experiment, task.variant_label, table_id, variable
    )
    if not fileset.files:
        return None

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
            return None

    n_workers = max(1, min(config.esgf.download_workers, len(fileset.files)))
    logging.info(
        "    downloading %d files (%.1f GB) for %s with %d workers ...",
        len(fileset.files),
        fileset.total_size / 1e9,
        variable,
        n_workers,
    )
    local_paths = _download_files(fileset.files, scratch, n_workers)

    logging.info("    opening and concatenating %s ...", variable)
    ds = _open_netcdf_files(local_paths, variable)
    ds = normalize_plev(ds)
    if table_id == "fx":
        if "time" in ds.dims:
            ds = ds.isel(time=0, drop=True)
    else:
        ds = apply_time_subset(ds, cfg)
        if ds.sizes.get("time", 0) == 0:
            return None
        ds, msg = resolve_time_duplicates(ds, variable, allow_dedupe=cfg.allow_dedupe)
        if msg:
            logging.warning("    %s", msg)
    return ds


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
    ds = _download_native_variable(task, variable, table_id, config, scratch)
    if ds is None:
        cleanup_variable_files(scratch, variable)
        return None, {}

    logging.info("    regridding %s ...", variable)
    regridded, methods = regrid_variables(ds, target_grid, cfg)
    regridded = regridded.load()

    cleanup_variable_files(scratch, variable)
    return regridded, methods


def _masked_regrid_3d_state(
    task: ESGFDatasetTask,
    config: ESGFProcessConfig,
    target_grid: xr.Dataset,
    scratch: Path,
    present_3d: list[str],
    *,
    make_regridder_fn=make_regridder,
    threshold: float = 0.5,
    time_chunk: int = 730,
) -> tuple[
    Optional[dict[str, xr.DataArray]],
    Optional[xr.DataArray],
    Optional[xr.DataArray],
    dict[str, str],
]:
    """Source-grid below-surface-masked regrid of the 3D plev state.

    Downloads the native 3D fields the model publishes (a subset of
    ``MASKED_3D_STATE_FIELDS`` — must include ``zg``) plus native ``orog``,
    computes the native ``zg < orog`` validity once, and returns the
    masked-regridded fields (plev-dimensioned, on the target grid), the shared
    target-grid per-level validity, the regridded surface height, and the
    per-field regrid methods.

    A single regridder built on the native grid serves both the valid-only
    (``skipna``) data regrid and the 0/1 indicator regrid — the native 3D
    fields and ``orog`` share one grid. ``make_regridder_fn`` is injectable so
    tests can supply a fake (block-mean) regridder without ESMF.

    Returns ``(None, None, None, {})`` when the state can't be masked (no
    ``zg``, no ``orog``, or nothing downloaded) — the caller then falls back to
    the unmasked per-variable path.
    """
    cfg = config.resolve(task.source_id, task.experiment, task.variant_label)
    if "zg" not in present_3d or not task.has_orog:
        return None, None, None, {}

    downloaded: list[str] = []
    orog_ds = _download_native_variable(task, "orog", "fx", config, scratch)
    if orog_ds is None or "orog" not in orog_ds:
        cleanup_variable_files(scratch, "orog")
        return None, None, None, {}
    downloaded.append("orog")
    orog_native = orog_ds["orog"]

    native: dict[str, xr.DataArray] = {}
    zg_native_ds: Optional[xr.Dataset] = None
    for v in present_3d:
        ds_v = _download_native_variable(
            task, v, cmip6_source_table(v), config, scratch
        )
        downloaded.append(v)
        if ds_v is not None and v in ds_v:
            native[v] = ds_v[v]
            if v == "zg":
                zg_native_ds = ds_v

    if "zg" not in native or zg_native_ds is None:
        for v in downloaded:
            cleanup_variable_files(scratch, v)
        return None, None, None, {}

    # Snap every field + orog onto zg's native horizontal grid. Across CMIP6
    # tables the same physical grid is stored with tiny float-rep differences in
    # its lat/lon coords (day ``zg`` vs fx ``orog``, and sometimes between day
    # fields); left as-is, xarray's implicit alignment silently INTERSECTS them
    # (e.g. 96 -> 56 lats), corrupting the mask and crashing the shared
    # regridder. A field on a genuinely different grid (shape/coords mismatch) is
    # data we can't source-grid mask -> drop it (and bail to the unmasked
    # fallback if it's orog or zg itself).
    ref_lat = native["zg"]["lat"]
    ref_lon = native["zg"]["lon"]
    orog_native = align_to_reference_grid(orog_native, ref_lat, ref_lon)
    if orog_native is None:
        logging.warning(
            "  orog is on a different horizontal grid than zg; cannot source-grid "
            "mask — falling back to the unmasked path"
        )
        for v in downloaded:
            cleanup_variable_files(scratch, v)
        return None, None, None, {}
    harmonized: dict[str, xr.DataArray] = {}
    for v, da in native.items():
        aligned = align_to_reference_grid(da, ref_lat, ref_lon)
        if aligned is None:
            logging.warning(
                "  %s is on a different horizontal grid than zg; dropping it from "
                "the source-grid masked 3D state (unusable data)",
                v,
            )
            continue
        harmonized[v] = aligned
    native = harmonized

    # Merge on the common TIME axis (lat/lon are now identical, so the inner
    # join only intersects timestamps — the shared native validity from zg then
    # lines up cell-for-cell with every field before the ``.where`` mask).
    merged_native = xr.merge(
        [native[v].rename(v) for v in native], compat="override", join="inner"
    )
    native_aligned = {v: merged_native[v] for v in native}
    zg_native = native_aligned["zg"]

    # Build the regridder from the native ``zg`` *dataset* (not the merged
    # DataArrays) so its lat/lon bounds survive — conservative regridding needs
    # them, and every 3D field + orog share this one native grid.
    method = cfg.regrid.method_for("zg")
    regridder, actual_method = make_regridder_fn(zg_native_ds, target_grid, method)

    def regrid_data(da: xr.DataArray) -> xr.DataArray:
        return regridder(da, keep_attrs=True, skipna=True)

    def regrid_indicator(da: xr.DataArray) -> xr.DataArray:
        return regridder(da, skipna=False)

    # Time-chunk the masked regrid. ``apply_time_subset`` fancy-indexes each
    # native field fully into RAM (~7.7 GB each at 96x192 x 8 plev x ~13k days),
    # so masking all five at once (plus orog/validity) OOM-killed the 32Gi pod.
    # Regridding one ``time_chunk``-day segment at a time and concatenating the
    # small target-grid results (45x90 at 4deg) bounds peak memory to a single
    # segment's natives regardless of record length or model. xesmf is not
    # thread-safe, so compute each segment under the synchronous scheduler; the
    # eager per-segment ``.load()`` also gives ``fill_below_surface_smooth``
    # (which writes into ``.values``) the numpy-backed fields it needs.
    import dask

    dask.config.set(scheduler="synchronous")

    n_time = int(zg_native.sizes["time"])
    n_seg = (n_time + time_chunk - 1) // time_chunk
    logging.info(
        "    masked 3D regrid: %d timesteps in %d segments of %d (rss %.0f MiB)",
        n_time,
        n_seg,
        time_chunk,
        rss_mib(),
    )
    regridded_parts: dict[str, list[xr.DataArray]] = {v: [] for v in native_aligned}
    valid_parts: list[xr.DataArray] = []
    for seg_i, start in enumerate(range(0, n_time, time_chunk)):
        sl = slice(start, start + time_chunk)
        seg_fields = {v: native_aligned[v].isel(time=sl) for v in native_aligned}
        seg_regridded, seg_valid = source_grid_masked_regrid(
            seg_fields,
            seg_fields["zg"],
            orog_native,
            regrid_data,
            regrid_indicator,
            threshold=threshold,
        )
        for v, da in seg_regridded.items():
            regridded_parts[v].append(da.load())
        valid_parts.append(seg_valid.load())
        logging.info(
            "    masked 3D segment %d/%d done (rss %.0f MiB)",
            seg_i + 1,
            n_seg,
            rss_mib(),
        )

    regridded_3d = {
        v: xr.concat(parts, dim="time") for v, parts in regridded_parts.items()
    }
    valid_target = xr.concat(valid_parts, dim="time")
    hgtsfc_target = regrid_data(orog_native).load()  # static, no time dim
    logging.info("    masked 3D regrid complete (rss %.0f MiB)", rss_mib())

    for v in downloaded:
        cleanup_variable_files(scratch, v)

    methods = {v: actual_method for v in native_aligned}
    return regridded_3d, valid_target, hgtsfc_target, methods


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

        # The 3D plev state ({ua,va,hus,zg,ta} the model publishes) goes
        # through source-grid below-surface masking (native zg<orog before
        # regrid); everything else is regridded per-variable, unmasked.
        masked_3d_present = [v for v in MASKED_3D_STATE_FIELDS if v in all_day_vars]
        generic_day_vars = [v for v in all_day_vars if v not in masked_3d_present]

        regridded_vars: dict[str, xr.Dataset] = {}
        for v in generic_day_vars:
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

        # Source-grid masked 3D state. Falls back to the legacy unmasked
        # per-variable path (+ the NaN-union below-surface mask at step 5/6)
        # when the model lacks orog or zg.
        masked_state_ds: Optional[xr.Dataset] = None
        mask_field_map: dict[str, str] = {}
        masked_field_names: set[str] = set()
        used_source_grid_masking = False
        if masked_3d_present:
            logging.info(
                "  [%s] source-grid masked 3D state: %s",
                task.source_id,
                masked_3d_present,
            )
            try:
                masked_3d, valid_target_3d, hgtsfc_3d, m3d_methods = (
                    _masked_regrid_3d_state(
                        task, config, target, scratch, masked_3d_present
                    )
                )
            except (SimulationBoundaryError, DuplicateTimestampsError) as e:
                row.status = "skipped"
                row.skip_reason = str(e)
                cleanup_scratch_dir(scratch)
                return row
            if masked_3d is not None:
                row.regrid_methods.update(m3d_methods)
                masked_field_names = set(masked_3d)
                logging.info(
                    "  masked 3D regrid returned; assembling (rss %.0f MiB)",
                    rss_mib(),
                )
                masked_state_ds, mask_field_map = assemble_masked_3d_state(
                    masked_3d, valid_target_3d, hgtsfc_3d
                )
                logging.info("  masked 3D state assembled (rss %.0f MiB)", rss_mib())
                used_source_grid_masking = True
                row.mask_source = "source_grid"
            else:
                row.warnings.append(
                    "source-grid masking unavailable (no orog/zg); "
                    "3D state regridded unmasked"
                )
                for v in masked_3d_present:
                    try:
                        result, methods = _download_and_regrid_variable(
                            task, v, cmip6_source_table(v), target, config, scratch
                        )
                        row.regrid_methods.update(methods)
                        if result is not None:
                            regridded_vars[v] = result
                    except (SimulationBoundaryError, DuplicateTimestampsError) as e:
                        row.status = "skipped"
                        row.skip_reason = str(e)
                        cleanup_scratch_dir(scratch)
                        return row

        present_core = set(regridded_vars) | masked_field_names
        missing_core = [v for v in cfg.core_variables if v not in present_core]
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

        # 2. Merge all regridded daily variables (+ the masked 3D state, which
        # already carries its filled fields, thicknesses, per-cell masks and
        # surface height).
        merge_inputs = list(regridded_vars.values())
        if masked_state_ds is not None:
            merge_inputs.append(masked_state_ds)
        day_regridded = xr.merge(merge_inputs, compat="override", join="inner")

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

        # 4. Static fields. When source-grid masking ran, the target-grid
        # surface height ``orog`` is already in ``day_regridded`` (regridded
        # with the same regridder as ``zg`` for a self-consistent thickness
        # anchor), so skip re-downloading it here.
        static_ds: Optional[xr.Dataset] = None
        for static_var in cfg.static_variables:
            if static_var == "orog" and used_source_grid_masking:
                continue
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

        # 5-6. Below-surface mask + smooth-flood fill of the level-valued 3D
        # state. Skipped when source-grid masking ran — those fields are
        # already masked-regridded, smooth-filled and carry per-cell masks;
        # this is the legacy fallback path (target-grid NaN-union mask) for
        # models processed without native zg/orog masking.
        stage_t0 = time.monotonic()
        if not used_source_grid_masking:
            orog: Optional[xr.DataArray] = None
            if static_ds is not None and "orog" in static_ds:
                orog = static_ds["orog"]
            mask, row.mask_source = compute_below_surface_mask(day_regridded, orog)
            if mask is not None:
                for v in ("ua", "va", "hus", "zg"):
                    if v in day_regridded:
                        day_regridded[v] = fill_below_surface_smooth(
                            day_regridded[v], mask
                        )
                day_regridded = day_regridded.assign(below_surface_mask=mask)
        _stage("below_surface_mask", stage_t0)

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

        # 12. Flatten plev, then stamp the per-cell ``mask_variable`` attrs on
        # the now-flattened 3D + thickness fields (the mapping is keyed by the
        # flattened names, e.g. ``ta1000`` -> ``mask_1000``).
        day_regridded = flatten_plev_variables(day_regridded)
        if used_source_grid_masking:
            attach_mask_attributes(day_regridded, mask_field_map)

        # 13. Harmonize temperatures to K (some CMIP6 publishers
        # emit ``tos``/``tob``/``sitemptop`` in °C). See process.py.
        # Skip mask channels — the ``_mask``-suffixed surface/ocean indicators
        # and the ``mask_``-prefixed per-cell loss/eval masks are 0/1 fields,
        # not temperatures.
        for v in list(day_regridded.data_vars):
            if v.endswith("_mask") or v.startswith("mask_"):
                continue
            da, msg = harmonize_temperature_to_kelvin(day_regridded[v], var_id=v)
            if msg:
                row.warnings.append(msg)
                if "converted" in msg:
                    day_regridded[v] = da

        # 14. Rename CMIP6 variables to the baseline convention
        # (see process.py for rationale).
        day_regridded = apply_output_renames(day_regridded, CMIP_TO_OUTPUT_RENAMES)

        # 15. Sanity checks — advisory only.
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


def _select_day_augmentables(
    optional_variables: list[str],
    available_day_variables: list[str],
    existing_vars: set[str],
    failed_augment_vars: set[str] | None = None,
) -> list[str]:
    """Return CMIP6 names of optional day-cadence variables to augment.

    Filtering rules:
    - The variable must be published for this (source, experiment,
      member) per ``available_day_variables`` (which already merges
      day-table + CFday-table entries).
    - The variable's renamed output name (via
      ``CMIP_TO_OUTPUT_RENAMES``) must not already be in
      ``existing_vars`` — augmenting an existing variable would
      conflict with the prior write.
    - The variable's renamed output name must not be in
      ``failed_augment_vars`` — a prior augment pass tried this
      variable and failed downstream of the network (regrid, write).
      Retrying would re-download the same files for the same
      deterministic failure. Drop the name from the sidecar's
      ``esgf_failed_augment_variables`` to force a retry.

    The actual download / regrid / write happens in
    :func:`_augment_day_variables` — this is the pure-function bit so
    we can unit-test the filtering without touching the network.

    Parameters:
        optional_variables: Configured optional day-cadence variables
            to consider (typically ``cfg.optional_variables``).
        available_day_variables: CMIP6 variable names the model
            actually publishes on either ``day`` or ``CFday`` for
            this (source, experiment, member). Comes from
            ``ESGFDatasetTask.available_day_variables``.
        existing_vars: Output names already present in the existing
            zarr (from ``DatasetIndexRow.variables_present``).
        failed_augment_vars: Output names a prior augment pass tried
            and failed on (from
            ``DatasetIndexRow.esgf_failed_augment_variables``).
            Defaults to empty for callers (e.g. unit tests) that
            don't track persisted failure state.

    Returns:
        CMIP6 names to download + augment, in deterministic order.
    """
    augmentable: list[str] = []
    available = set(available_day_variables)
    failed = failed_augment_vars or set()
    for var in optional_variables:
        if var not in available:
            continue
        out_name = CMIP_TO_OUTPUT_RENAMES.get(var, var)
        if out_name in existing_vars:
            continue
        if out_name in failed:
            continue
        augmentable.append(var)
    return augmentable


def _augment_day_variables(
    task: ESGFDatasetTask,
    config: ESGFProcessConfig,
    cfg,
    zarr_path: str,
    target_grid: xr.Dataset,
    existing_vars: set[str],
    existing_row: DatasetIndexRow,
    scratch: Path,
) -> list[str]:
    """Augment the existing zarr with missing day-cadence variables.

    Counterpart to the surface-and-ocean augment loop, focused on
    optional day / CFday day-cadence variables that Pangeo's CMIP6
    mirror misses for most models (clear-sky + TOA radiation,
    wap500, clivi, clwvi, ta700, surface pressure). ESGF carries
    these for 200-350 (source, experiment, member) tuples; Pangeo
    publishes only 1-3 each, so the augment recovers near-cohort
    coverage.

    Each augmentable variable goes through the same pipeline as the
    fresh-process path (download → regrid → rename → harmonize →
    write); only surface-and-ocean masking / monthly_causal mapping
    is skipped (these are real daily atmospheric variables).

    Returns the list of output names successfully added.
    """
    augmentable = _select_day_augmentables(
        cfg.optional_variables,
        task.available_day_variables,
        existing_vars,
        set(existing_row.esgf_failed_augment_variables),
    )
    if not augmentable:
        return []

    logging.info(
        "  augmenting with %d day-cadence variables: %s",
        len(augmentable),
        ", ".join(augmentable),
    )

    added: list[str] = []
    for var in augmentable:
        # Bound outside the try so the except handler can record the
        # failure under the canonical output name.
        out_name = CMIP_TO_OUTPUT_RENAMES.get(var, var)
        try:
            source_table = cmip6_source_table(var)
            logging.info(
                "  [%s] augment day %s/%s -> %s ...",
                task.source_id,
                source_table,
                var,
                out_name,
            )
            result, methods = _download_and_regrid_variable(
                task, var, source_table, target_grid, config, scratch
            )
            existing_row.regrid_methods.update(
                {out_name if k == var else k: v for k, v in methods.items()}
            )
            if result is None:
                continue
            # Rename to the output convention BEFORE harmonize so the
            # warning messages reference the canonical output name.
            renamed = apply_output_renames(result, CMIP_TO_OUTPUT_RENAMES)
            harmonized: dict[str, xr.DataArray] = {}
            for name, da in renamed.data_vars.items():
                converted, msg = harmonize_temperature_to_kelvin(da, var_id=name)
                if msg:
                    logging.info("  augment harmonize: %s", msg)
                    existing_row.warnings.append(msg)
                harmonized[name] = converted
            new_ds = xr.Dataset(harmonized)
            new_ds.to_zarr(
                zarr_path,
                mode="a",
                consolidated=False,
                zarr_format=3,
                align_chunks=True,
            )
            added.extend(new_ds.data_vars.keys())
        except Exception as e:  # noqa: BLE001
            logging.warning(
                "  augment day %s failed: %s: %s — skipping this variable",
                var,
                type(e).__name__,
                e,
            )
            existing_row.warnings.append(
                f"augment day {var} failed: {type(e).__name__}: {e}"
            )
            if out_name not in existing_row.esgf_failed_augment_variables:
                existing_row.esgf_failed_augment_variables.append(out_name)
    return added


def _should_derive_total_water_path(
    day_added: list[str],
    existing_vars: set[str],
    added_names: list[str],
) -> bool:
    """True iff the augment pass should derive ``total_water_path``.

    Derivation requires ``water_vapor_path`` + ``clwvi`` both to be
    on disk after the augment writes finish, AND for ``clwvi`` to
    have been added by *this* pass (otherwise total_water_path
    either already exists or the prior pass deliberately didn't
    derive it). ``water_vapor_path`` can come from either source:

    - Pre-augment (in ``existing_vars``): the Pangeo zarr already
      carried Eday.prw.
    - Same pass (in ``added_names``): the surface-and-ocean loop
      added it via ESGF earlier in this run.

    The historical bug only checked ``existing_vars``, missing
    the common v2 case where both inputs are augmented in the same
    pass (22 of 26 eligible datasets affected).
    """
    if "clwvi" not in day_added:
        return False
    return "water_vapor_path" in existing_vars or "water_vapor_path" in added_names


def augment_one_esgf(
    task: ESGFDatasetTask,
    config: ESGFProcessConfig,
    existing_row: DatasetIndexRow,
    *,
    retry_failed_augments: bool = False,
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

    Day-cadence augment (CFday + day): a second pass picks up
    optional day-table + CFday-table variables Pangeo missed (clear-
    sky + TOA radiation, wap500, clivi, clwvi, ta700, surface
    pressure). Same download → regrid → rename → harmonize → write
    flow as the surface-and-ocean loop, without the mask/fill step.

    Limitations (v2):
    - 3D plev day variables (ua, va, hus, zg) are still skipped —
      augmenting them requires running the flatten step against the
      existing zarr's plev convention.
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

    # ``retry_failed_augments``: wipe the prior failure list so
    # previously-failed variables become eligible again. Whatever
    # fails this pass gets re-added to the (now empty) list, so
    # ``esgf_failed_augment_variables`` after the run reflects only
    # variables that failed *this* attempt — not a stale accumulation.
    if retry_failed_augments:
        existing_row.esgf_failed_augment_variables = []
    existing_vars = set(existing_row.variables_present)
    failed_aug_vars = set(existing_row.esgf_failed_augment_variables)
    augmentable: list = []
    for h in SURFACE_AND_OCEAN_VARIABLES:
        if h.output_name not in cfg.surface_and_ocean_variables:
            continue
        if h.output_name not in task.available_surface_and_ocean_variables:
            continue
        if h.output_name in existing_vars:
            continue
        # Prior augment pass already tried + failed this variable.
        # Re-attempting would redownload the same files for the same
        # deterministic failure (typically ESMF regrid rc=506 from a
        # publisher grid mismatch). Force-retry by deleting the entry
        # from the sidecar's ``esgf_failed_augment_variables``.
        if h.output_name in failed_aug_vars:
            continue
        augmentable.append(h)

    day_augmentable = _select_day_augmentables(
        cfg.optional_variables,
        task.available_day_variables,
        existing_vars,
        failed_aug_vars,
    )
    if not augmentable and not day_augmentable:
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
    # ``consolidated=False`` (not True): a handful of v2 zarrs from the
    # initial ingest don't have consolidated metadata (their ingest pod
    # was preempted between the last to_zarr write and the
    # ``consolidate_metadata`` call). The augment reconsolidates at the
    # end (see line ~1183), so opening here without consolidated
    # metadata is a safe one-time slowdown; opening with
    # ``consolidated=True`` would crash those zarrs with
    # ``ValueError: Consolidated metadata requested ... but not found``.
    existing_ds = xr.open_zarr(zarr_path, consolidated=False)
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
                # Apply temperature unit harmonization — the fresh
                # process_one_esgf path runs this on every non-mask
                # variable but the augment path used to skip it,
                # which let 77 ESGF-augmented ``omon_tob`` fields
                # ship in degC and land in the zarr as ~3 K nonsense.
                # Pass ``var_id=name`` so the spec-default lookup
                # also fires for sources where the ``units`` attr is
                # absent.
                harmonized: dict[str, xr.DataArray] = {}
                for name, da in outputs.items():
                    if name.endswith("_mask"):
                        harmonized[name] = da
                        continue
                    converted, msg = harmonize_temperature_to_kelvin(da, var_id=name)
                    if msg:
                        logging.info("  augment harmonize: %s", msg)
                        existing_row.warnings.append(msg)
                    harmonized[name] = converted
                new_ds = xr.Dataset(harmonized)
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
                if h.output_name not in existing_row.esgf_failed_augment_variables:
                    existing_row.esgf_failed_augment_variables.append(h.output_name)
        existing_ds.close()

        # Day-cadence augment (CFday + day). Runs after the
        # surface-and-ocean loop so existing_vars reflects any
        # surface-and-ocean adds — though in practice the two
        # variable sets don't intersect, this ordering keeps the
        # filter logic clean.
        day_added = _augment_day_variables(
            task,
            config,
            cfg,
            zarr_path,
            target,
            existing_vars | set(added_names),
            existing_row,
            scratch,
        )
        added_names.extend(day_added)

        # Derived total_water_path mirrors the fresh-process path's
        # step 9_pre — see :func:`_should_derive_total_water_path`
        # for the predicate.
        if _should_derive_total_water_path(
            day_added=day_added,
            existing_vars=existing_vars,
            added_names=added_names,
        ):
            try:
                reopened = xr.open_zarr(zarr_path, consolidated=False)
                twp = compute_total_water_path(
                    reopened["water_vapor_path"], reopened["clwvi"]
                )
                twp_ds = xr.Dataset({"total_water_path": twp})
                twp_ds.to_zarr(
                    zarr_path,
                    mode="a",
                    consolidated=False,
                    zarr_format=3,
                    align_chunks=True,
                )
                reopened.close()
                added_names.append("total_water_path")
                logging.info("  derived total_water_path from water_vapor_path + clwvi")
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "  derive total_water_path failed: %s: %s — skipping",
                    type(e).__name__,
                    e,
                )
                existing_row.warnings.append(
                    f"derive total_water_path failed: {type(e).__name__}: {e}"
                )
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
    retry_failed_augments: bool = False,
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
                row = augment_one_esgf(
                    task,
                    config,
                    existing,
                    retry_failed_augments=retry_failed_augments,
                )
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
    parser.add_argument(
        "--retry-failed-augments",
        action="store_true",
        help=(
            "For each dataset's augment pass: clear the sidecar's "
            "``esgf_failed_augment_variables`` before deciding what to "
            "augment, so variables a prior pass tried + failed become "
            "eligible again. Use after fixing the upstream issue (e.g. "
            "an ESMF regrid failure that's been resolved by a config "
            "or library update). Variables that fail again get re-added "
            "to the sidecar's failure list."
        ),
    )
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

    rows = run(
        config,
        tasks,
        force=args.force,
        retry_failed_augments=args.retry_failed_augments,
    )

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
