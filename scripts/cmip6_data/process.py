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
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import ProcessConfig, ResolvedDatasetConfig, make_label  # noqa: E402
from grid import make_target_grid  # noqa: E402
from index import DatasetIndexRow, write_index, write_sidecar  # noqa: E402

# Physics constants for hypsometric layer-mean T derivation.
R_D = 287.05  # J / (kg K), dry-air gas constant
G = 9.80665  # m / s^2, standard gravity
EPS = 0.608  # (R_v / R_d) - 1, for virtual-to-actual temperature


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


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _open_zstore(url: str) -> xr.Dataset:
    """Open a Pangeo GCS zarr. Consolidated metadata; don't decode times
    yet — we'll do that explicitly after harmonising coords.
    """
    mapper = fsspec.get_mapper(url)
    return xr.open_zarr(mapper, consolidated=True, decode_times=True)


_EXPECTED_MEAN_CELL_METHODS = re.compile(r"time:\s*mean")


def _validate_cell_methods(ds: xr.Dataset, variables: list[str]) -> list[str]:
    """Return the subset of ``variables`` whose ``cell_methods`` attr
    does not contain ``time: mean``. Variables missing from ``ds`` are
    silently ignored here — absence is handled upstream.
    """
    mismatches = []
    for v in variables:
        if v not in ds.data_vars:
            continue
        cm = str(ds[v].attrs.get("cell_methods", ""))
        if not _EXPECTED_MEAN_CELL_METHODS.search(cm):
            mismatches.append(v)
    return mismatches


def _make_regridder(source_ds: xr.Dataset, target: xr.Dataset, method: str):
    """Build an xESMF regridder, importing lazily so this module can be
    loaded in envs without xESMF (e.g. unit tests on the selection logic).
    """
    import xesmf

    return xesmf.Regridder(source_ds, target, method, periodic=True)


def _regrid_variables(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    cfg: ResolvedDatasetConfig,
) -> tuple[xr.Dataset, dict[str, str]]:
    """Regrid all data variables in ``ds`` to ``target_grid``. Returns
    the regridded dataset and a dict ``{variable: method}``.
    """
    method_for = cfg.regrid.method_for
    # Group variables by method so we make one Regridder per method.
    by_method: dict[str, list[str]] = {}
    for v in ds.data_vars:
        by_method.setdefault(method_for(v), []).append(v)

    pieces = []
    used: dict[str, str] = {}
    for method, vars_ in by_method.items():
        sub = ds[vars_]
        regridder = _make_regridder(sub, target_grid, method)
        pieces.append(regridder(sub, keep_attrs=True))
        used.update({v: method for v in vars_})

    regridded = xr.merge(pieces)
    # Preserve non-horizontal coords (time, plev, etc.) that xESMF strips.
    for coord in ds.coords:
        if coord not in regridded.coords and ds[coord].ndim <= 1:
            regridded = regridded.assign_coords({coord: ds[coord]})
    return regridded, used


_PLEV8_DEFAULT_HPA = np.array([1000, 850, 700, 500, 250, 100, 50, 10], dtype=np.float64)


def _normalize_plev(ds: xr.Dataset) -> xr.Dataset:
    """Ensure plev axis is descending-in-altitude: index 0 is the lowest
    pressure level (= 1000 hPa), index 7 is the highest (= 10 hPa).
    CMIP6 publishes plev in Pa with either ascending or descending
    order; we normalise to descending-pressure.
    """
    if "plev" not in ds.dims:
        return ds
    plev = ds["plev"].values
    if len(plev) > 1 and plev[0] < plev[-1]:
        ds = ds.isel(plev=slice(None, None, -1))
    return ds


def _compute_below_surface_mask(
    ds: xr.Dataset,
    orog: Optional[xr.DataArray],
) -> tuple[xr.DataArray, str]:
    """Return (mask, source) where mask is a time-varying uint8 array
    with dims (time, plev, lat, lon). ``source`` is one of
    ``nan_union`` or ``orog_static``.
    """
    three_d = [v for v in ("ua", "va", "hus", "zg") if v in ds.data_vars]
    if three_d:
        nan_union = ds[three_d[0]].isnull()
        for v in three_d[1:]:
            nan_union = nan_union | ds[v].isnull()
        if bool(nan_union.any()):
            return nan_union.astype("uint8").rename("below_surface_mask"), "nan_union"

    if orog is None:
        raise RuntimeError(
            "Cannot derive below_surface_mask: no NaN pattern and no orog."
        )
    # zg is time-varying (altitude of pressure level varies with synoptic
    # state); orog is static. Broadcasting makes the result time-varying.
    mask = (ds["zg"] < orog).astype("uint8").rename("below_surface_mask")
    return mask, "orog_static"


def _nearest_above_fill(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Fill below-surface cells in ``da`` with the value at the lowest
    above-surface level in that column. Works for any number of
    consecutive masked bottom levels.

    ``da`` is (time, plev, lat, lon); ``mask`` is uint8 same shape.
    Plev axis is assumed descending in altitude (index 0 = 1000 hPa).
    """
    filled = da.where(mask == 0)
    filled = filled.bfill("plev")
    return filled


def _compute_derived_layer_T(ds: xr.Dataset) -> xr.Dataset:
    """Add ``ta_derived_layer_{0..N-1}`` from zg + hus + plev via the
    hypsometric equation. One layer value per gap between adjacent
    plev levels, assigned to the log-pressure midpoint.
    """
    plev_pa = ds["plev"].values  # Pa
    z = ds["zg"]  # m
    q = ds["hus"]  # kg/kg
    out = {}
    n_layers = len(plev_pa) - 1
    for i in range(n_layers):
        p_lo = plev_pa[i]  # higher pressure (if descending in altitude)
        p_hi = plev_pa[i + 1]
        dz = z.isel(plev=i + 1) - z.isel(plev=i)
        q_mean = 0.5 * (q.isel(plev=i) + q.isel(plev=i + 1))
        tv = G * dz / (R_D * np.log(p_lo / p_hi))
        t = tv / (1.0 + EPS * q_mean)
        p_lo_hpa = p_lo / 100
        p_hi_hpa = p_hi / 100
        t.attrs = {
            "long_name": (
                f"derived layer-mean T between "
                f"{p_lo_hpa:.0f} and {p_hi_hpa:.0f} hPa"
            ),
            "units": "K",
            "derivation": "hypsometric from zg + hus",
        }
        out[f"ta_derived_layer_{i}"] = t.drop_vars("plev", errors="ignore")
    return ds.assign(**out)


def _interp_monthly_to_daily(
    monthly: xr.DataArray,
    daily_time: xr.DataArray,
    method: str,
) -> xr.DataArray:
    """Interpolate monthly values onto the daily axis; constant-value
    extrapolation at the start and end of the series (daily stamps
    outside the first / last monthly bracket take the nearest monthly
    value).
    """
    interp = monthly.interp(time=daily_time, method=method)
    # bfill handles leading NaN (days before the first monthly point);
    # ffill handles trailing NaN (days after the last monthly point).
    return interp.bfill("time").ffill("time")


def _apply_time_subset(ds: xr.Dataset, cfg: ResolvedDatasetConfig) -> xr.Dataset:
    window = cfg.time_subset.get(cfg.experiment)
    if window is None:
        return ds
    return ds.sel(time=slice(window.start, window.end))


_STALE_ENCODING_KEYS = (
    # zarr v2 codec metadata that zarr v3 can't consume — strip before
    # writing to avoid silent misinterpretation.
    "compressors",
    "filters",
    "preferred_chunks",
    # Source chunk shapes; we're re-chunking, so don't carry the old.
    "chunks",
    "shards",
    # CMIP6 convention: 1e+20 as sentinel fill. zarr v3 floats store NaN
    # natively; keep it simple and let xarray write NaN as NaN.
    "_FillValue",
    "missing_value",
    "dtype",
)


def _clear_stale_encoding(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    for var in {**ds.coords, **ds.data_vars}.values():
        for k in _STALE_ENCODING_KEYS:
            var.encoding.pop(k, None)
    return ds


def _write_zarr(ds: xr.Dataset, path: str, cfg: ResolvedDatasetConfig) -> None:
    """Write ``ds`` to ``path`` with zarr v3 chunks + shards per the
    config. time dim uses (chunk_time, shard_time); other dims are
    single chunk / single shard (full extent).
    """
    ds = _clear_stale_encoding(ds)

    # Materialize the dataset before writing. The xesmf Regridder wraps
    # the ESMF C library, which is not thread-safe; leaving the regrid
    # lazy in the dask graph lets dask parallelise per-chunk, and the
    # resulting concurrent Regridder.__call__ invocations can silently
    # produce NaN for some chunks. Computing up front runs the regrid
    # once, sequentially, and gives deterministic output.
    logging.info("  materializing dataset in memory before write...")
    ds = ds.load()

    chunk_time = cfg.chunking.chunk_time
    shard_time = cfg.chunking.shard_time

    encoding: dict[str, dict] = {}
    for v in list(ds.data_vars) + list(ds.coords):
        var = ds[v]
        chunks = []
        shards: list[int] = []
        for dim, size in zip(var.dims, var.shape):
            if dim == "time":
                chunks.append(min(chunk_time, size))
                shards.append(min(shard_time or size, size))
            else:
                chunks.append(size)
                shards.append(size)
        enc: dict = {"chunks": tuple(chunks)}
        if shard_time is not None and "time" in var.dims:
            enc["shards"] = tuple(shards)
        encoding[v] = enc

    ds.to_zarr(path, mode="w", encoding=encoding, consolidated=True, zarr_format=3)


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

        # 2. Open day-table vars (core + optional if present).
        all_day_vars = cfg.core_variables + [
            v for v in cfg.optional_variables if v in day_zs
        ]
        opened: list[xr.Dataset] = []
        for v in all_day_vars:
            opened.append(_open_zstore(day_zs[v])[[v]])
        day_ds = xr.merge(opened, compat="override")

        # 3. Cell-methods validation.
        row.cell_methods_mismatch = _validate_cell_methods(day_ds, all_day_vars)
        if row.cell_methods_mismatch:
            row.warnings.append(
                f"cell_methods != 'time: mean' for {row.cell_methods_mismatch}"
            )

        # 4. Capture native calendar.
        if "time" in day_ds.dims:
            row.native_calendar = str(day_ds["time"].dt.calendar)

        # 5. Normalise plev ordering so index 0 is closest to the surface.
        day_ds = _normalize_plev(day_ds)

        # 6. Time-subset now (before regrid) to avoid regridding data
        #    we're going to throw away.
        day_ds = _apply_time_subset(day_ds, cfg)
        if day_ds.sizes.get("time", 0) == 0:
            row.status = "skipped"
            row.skip_reason = "no timesteps in configured time_subset"
            return row

        # 7. Regrid day variables to the target grid.
        target = make_target_grid(cfg.target_grid.name)
        day_regridded, methods_used = _regrid_variables(day_ds, target, cfg)
        row.regrid_methods.update(methods_used)

        # 8. Static fields from fx — regrid once, drop time.
        static_ds: Optional[xr.Dataset] = None
        fx_zs = task.zstores.get("fx", {})
        fx_have = [v for v in cfg.static_variables if v in fx_zs]
        if fx_have:
            fx_pieces = [
                _open_zstore(fx_zs[v])[[v]].squeeze(drop=True) for v in fx_have
            ]
            fx_merged = xr.merge(fx_pieces, compat="override")
            static_regridded, static_methods = _regrid_variables(fx_merged, target, cfg)
            static_ds = static_regridded
            row.regrid_methods.update(static_methods)

        # 9. Below-surface mask + nearest-above fill.
        orog: Optional[xr.DataArray] = None
        if static_ds is not None and "orog" in static_ds:
            orog = static_ds["orog"]
        mask, row.mask_source = _compute_below_surface_mask(day_regridded, orog)
        for v in ("ua", "va", "hus", "zg"):
            if v in day_regridded:
                day_regridded[v] = _nearest_above_fill(day_regridded[v], mask)
        day_regridded = day_regridded.assign(below_surface_mask=mask)

        # 10. Derived layer-mean T from filled zg + hus.
        day_regridded = _compute_derived_layer_T(day_regridded)

        # 11. Monthly forcings -> daily.
        for table, var in (("Amon", "ts"), ("SImon", "siconc")):
            z = task.zstores.get(table, {}).get(var)
            if not z:
                row.warnings.append(f"missing forcing {var} from {table}")
                continue
            monthly = _open_zstore(z)[[var]]
            monthly = _apply_time_subset(monthly, cfg)
            monthly_r, methods_used = _regrid_variables(monthly, target, cfg)
            row.regrid_methods.update(methods_used)
            interp = _interp_monthly_to_daily(
                monthly_r[var], day_regridded["time"], cfg.forcing_interpolation
            )
            day_regridded[var] = interp

        # 11b. Sea-ice fraction is only defined over ocean cells — emit
        # a time-invariant ``siconc_mask`` from the regridded ocean-grid
        # coverage, then replace the residual land NaN with 0 so the
        # stored data is NaN-free. The mask is a 2D (lat, lon) uint8
        # (1 = valid / ocean, 0 = land or missing source coverage); we
        # take it from the first timestep, since the valid-cell pattern
        # is time-invariant for sea-ice fraction.
        if "siconc" in day_regridded:
            valid = (~day_regridded["siconc"].isel(time=0).isnull()).astype("uint8")
            day_regridded = day_regridded.assign(
                siconc_mask=valid.rename("siconc_mask")
            )
            day_regridded["siconc"] = day_regridded["siconc"].fillna(0.0)

        # 12. Attach static fields (broadcast along time implicitly at read).
        if static_ds is not None:
            for v in static_ds.data_vars:
                day_regridded[v] = static_ds[v]

        # 13. Sanity count of NaN cells in the filled 3D state. Should
        # be zero; any non-zero is a sign the fill logic missed
        # something. Uses eager .compute() because dask arrays don't
        # support .item().
        nan_total = 0
        for v in ("ua", "va", "hus", "zg"):
            if v in day_regridded:
                nan_total += int(day_regridded[v].isnull().sum().compute())
        row.n_nan_input_cells = nan_total

        # 14. Populate time metadata.
        row.n_timesteps = int(day_regridded.sizes.get("time", 0))
        if row.n_timesteps:
            row.time_start = str(day_regridded["time"].values[0])
            row.time_end = str(day_regridded["time"].values[-1])

        # 15. Write zarr + record variables.
        day_regridded.attrs["label"] = label
        day_regridded.attrs["source_id"] = task.source_id
        day_regridded.attrs["experiment"] = task.experiment
        day_regridded.attrs["variant_label"] = task.variant_label
        _write_zarr(day_regridded, zarr_path, cfg)
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
        "--max-datasets",
        type=int,
        default=None,
        help="Limit to this many datasets (after selection). Debug aid.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = ProcessConfig.from_file(args.config)
    if args.source_ids is not None:
        config.selection.source_ids = args.source_ids

    inventory = _load_inventory(config.inventory_path)
    tasks = select_datasets(inventory, config)
    if args.max_datasets is not None:
        tasks = tasks[: args.max_datasets]

    logging.info("Selected %d tasks.", len(tasks))

    if args.dry_run:
        for t in tasks:
            print(f"{t.source_id}\t{t.experiment}\t{t.variant_label}")
        return

    rows = run(config, tasks, force=args.force)
    write_index(rows, config.output_directory)

    # Final status summary.
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    logging.info("Done. Status breakdown: %s", by_status)


if __name__ == "__main__":
    main()
