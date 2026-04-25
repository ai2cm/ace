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


class _SimulationBoundaryError(ValueError):
    """Raised when duplicate time indices carry materially different
    data — a strong signal that two stitched simulations (e.g., a
    historical run and its ssp585 continuation) were concatenated into
    a single zarr without care. Silently deduplicating would hide a
    real discontinuity, so we stop and surface the issue.
    """


def _resolve_time_duplicates(ds: xr.Dataset, var_name: str) -> tuple[xr.Dataset, str]:
    """Detect duplicate timestamps and decide how to handle them.

    Returns ``(ds, message)`` where ``ds`` is the cleaned dataset and
    ``message`` is a non-empty warning string when duplicates were
    found and safely deduplicated (every duplicate pair was
    data-identical — the classic CMIP file-splice redundancy). If the
    duplicate timestamps carry *different* data, raise
    ``_SimulationBoundaryError`` so the caller skips the dataset; a
    real simulation boundary needs to be split into two separate
    zarr stores, not silently merged.
    """
    if "time" not in ds.dims:
        return ds, ""
    times = ds["time"].values
    if len(times) == np.unique(times).size:
        return ds, ""

    # Find duplicated timestamps.
    vals, counts = np.unique(times, return_counts=True)
    dup_times = vals[counts > 1]
    n_dup = int(counts[counts > 1].sum() - len(dup_times))

    da = ds[var_name]
    for t in dup_times:
        idxs = np.where(times == t)[0]
        first = da.isel(time=idxs[0]).load()
        for i in idxs[1:]:
            other = da.isel(time=i).load()
            # ``equal_nan=True`` so matching NaN patterns don't count
            # as a difference. Generous rtol for float rounding across
            # duplicated file writes.
            if not np.allclose(first, other, equal_nan=True, rtol=1e-5, atol=0):
                raise _SimulationBoundaryError(
                    f"duplicate time {t} in {var_name} has materially "
                    "different data across copies — looks like a "
                    "simulation-boundary stitch. Republish this "
                    "dataset with the two halves in separate stores "
                    "before ingesting."
                )

    # All duplicate pairs are data-identical → safe to dedupe.
    return (
        ds.drop_duplicates("time", keep="first"),
        f"{var_name}: {n_dup} duplicate timestamp(s) detected "
        "(data-identical, safely deduplicated)",
    )


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


def _normalize_regrid_source(ds: xr.Dataset) -> xr.Dataset:
    """Prepare ``ds`` so xesmf's conservative regridder can ingest it.

    Two things happen:

    * **Rename CMIP6 bounds to xesmf's names.** xesmf looks for
      ``lon_b`` / ``lat_b``; CMIP6 publishes ``lon_bnds`` / ``lat_bnds``
      on rectilinear grids and ``vertices_longitude`` /
      ``vertices_latitude`` on curvilinear ocean grids (tripolar etc.).
    * **Convert CF-style (N, M, 4) vertex bounds to xesmf's (N+1, M+1)
      corner mesh.** CMIP6 2D bounds store four corners per cell
      (counterclockwise from bottom-left); xesmf wants shared-corner
      arrays that are one larger in each direction. Uses
      ``cf_xarray.bounds_to_vertices``.
    """
    import cf_xarray  # noqa: F401  (registers the bounds helper)
    from cf_xarray import bounds_to_vertices

    ds = ds.copy()
    for src, dst in (
        ("vertices_longitude", "lon_b"),
        ("vertices_latitude", "lat_b"),
        ("lon_bnds", "lon_b"),
        ("lat_bnds", "lat_b"),
    ):
        if src in ds.variables and dst not in ds.variables:
            ds = ds.rename({src: dst})

    for name in ("lon_b", "lat_b"):
        if name not in ds.variables:
            continue
        da = ds[name]
        # If already 1D (N+1,) or 2D (N+1, M+1) in corner form,
        # leave it alone. If it's a CF-style (N, 2) or (N, M, 4), convert.
        if da.ndim == 3 and da.shape[-1] == 4:
            bounds_dim = da.dims[-1]
            ds = ds.drop_vars(name).assign_coords(
                {name: bounds_to_vertices(da, bounds_dim=bounds_dim)}
            )
        elif da.ndim == 2 and da.shape[-1] == 2:
            bounds_dim = da.dims[-1]
            ds = ds.drop_vars(name).assign_coords(
                {name: bounds_to_vertices(da, bounds_dim=bounds_dim)}
            )

    return ds


def _make_regridder(source_ds: xr.Dataset, target: xr.Dataset, method: str):
    """Build an xESMF regridder, importing lazily so this module can be
    loaded in envs without xESMF (e.g. unit tests on the selection logic).
    """
    import xesmf

    source_ds = _normalize_regrid_source(source_ds)
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

    Must be called on *un-filled* zg / hus. Running this on nearest-
    above-filled zg would force ``dz = 0`` below surface and collapse
    the derived T to zero; see ``_fill_derived_layer_T`` for the
    post-derivation fill.
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


def _fill_derived_layer_T(ds: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """Cascading nearest-above fill for ``ta_derived_layer_*``.

    Layer ``i`` is treated as invalid where either bounding level
    (``plev[i]`` or ``plev[i+1]``) is below-surface per ``mask``, since
    the layer-mean hypsometric formula uses the model's below-surface
    extrapolation there and produces unphysical values.

    The fill cascades top-down: layer N-2 is always valid (stratosphere);
    each layer ``i < N-2`` inherits layer ``i+1`` where invalid. Runs
    once per layer, so consecutive masked bottom layers all resolve to
    the topmost valid layer's value.
    """
    layer_vars = sorted(v for v in ds.data_vars if v.startswith("ta_derived_layer_"))
    n_layers = len(layer_vars)
    if n_layers < 2:
        return ds
    for i in range(n_layers - 2, -1, -1):
        layer_mask = (mask.isel(plev=i) | mask.isel(plev=i + 1)).astype(bool)
        this_var = f"ta_derived_layer_{i}"
        above_var = f"ta_derived_layer_{i + 1}"
        ds[this_var] = ds[this_var].where(~layer_mask, ds[above_var])
    return ds


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


_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _clip_date_for_calendar(date_str: str, calendar: str) -> str:
    """Clip a ``YYYY-MM-DD`` string to a valid day in the target
    calendar. Mostly matters for ``360_day``, where every month has
    exactly 30 days — so ``2010-12-31`` is an invalid date and would
    raise in ``xr.Dataset.sel(time=slice(...))``.

    For ``noleap`` Feb 29 is invalid but we don't currently produce
    that date in our configs; the 360_day case is the common one.
    """
    if calendar != "360_day":
        return date_str
    m = _ISO_DATE_RE.match(date_str)
    if not m:
        return date_str
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    d = min(d, 30)
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _apply_time_subset(ds: xr.Dataset, cfg: ResolvedDatasetConfig) -> xr.Dataset:
    window = cfg.time_subset.get(cfg.experiment)
    if window is None:
        return ds
    if "time" in ds.dims:
        try:
            calendar = str(ds["time"].dt.calendar)
        except (AttributeError, TypeError):
            calendar = "standard"
    else:
        calendar = "standard"
    start = _clip_date_for_calendar(window.start, calendar)
    end = _clip_date_for_calendar(window.end, calendar)
    return ds.sel(time=slice(start, end))


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


# Per-variable (min, max) physical bounds. Values outside these get
# flagged as sanity-check warnings. Loose bounds chosen to catch real
# bugs (wrong units, wrong sign, wrong derivation) without firing on
# (a) normal climatological extremes or (b) sub-epsilon floating-point
# residuals from conservative regridding — whose output we trust to
# preserve the area-weighted integral, so clipping would break that.
# Variables not listed here are not checked.
_EPS = 0.01  # tolerance for conservative-regrid rounding noise

_SANITY_RANGES: dict[str, tuple[float, float]] = {
    # Winds (m/s)
    "ua": (-200.0, 200.0),
    "va": (-200.0, 200.0),
    "uas": (-100.0, 100.0),
    "vas": (-100.0, 100.0),
    "sfcWind": (-_EPS, 100.0),
    # Specific humidity (kg/kg) — non-negative
    "hus": (-_EPS, 0.05),
    "huss": (-_EPS, 0.05),
    # Temperatures (K)
    "ts": (180.0, 340.0),
    "tas": (180.0, 340.0),
    # Pressure (Pa)
    "psl": (8.0e4, 1.1e5),
    # Precipitation rate (kg/m2/s) — daily means rarely exceed 1 mm/s
    "pr": (-_EPS, 0.01),
    # Sea-ice fraction — CMIP6 siconc is percent
    "siconc": (-_EPS, 100.0 + _EPS),
    # Radiation fluxes (W/m2) — rough global bounds
    "rsdt": (-_EPS, 600.0),
    "rsut": (-_EPS, 600.0),
    "rlut": (-_EPS, 400.0),
    "rsds": (-_EPS, 600.0),
    "rsus": (-_EPS, 600.0),
    "rlds": (-_EPS, 600.0),
    "rlus": (-_EPS, 700.0),
    # Turbulent fluxes (W/m2) — tropics + downward storms can reach
    # ~1000 at daily mean. Widened vs the v0 bounds.
    "hfss": (-1000.0, 1000.0),
    "hfls": (-500.0, 1200.0),
    # Land / ocean fractions (percent)
    "sftlf": (-_EPS, 100.0 + _EPS),
    # Orography (m)
    "orog": (-500.0, 9000.0),
}

# Derived layer-mean temperatures get a tighter physical bound.
_DERIVED_T_RANGE = (150.0, 350.0)


def _run_sanity_checks(ds: xr.Dataset) -> list[str]:
    """Run cheap per-variable range checks and a tas-vs-derived-T0
    sanity comparison. Returns a list of human-readable warnings; an
    empty list means all checks passed.
    """
    messages: list[str] = []

    for var, (lo, hi) in _SANITY_RANGES.items():
        if var not in ds.data_vars:
            continue
        arr = ds[var]
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmin < lo or vmax > hi:
            messages.append(
                f"{var} out of expected range [{lo}, {hi}]: "
                f"min={vmin:.3g}, max={vmax:.3g}"
            )

    lo, hi = _DERIVED_T_RANGE
    for i in range(8):
        var = f"ta_derived_layer_{i}"
        if var not in ds.data_vars:
            continue
        arr = ds[var]
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmin < lo or vmax > hi:
            messages.append(
                f"{var} out of range [{lo}, {hi}] K: " f"min={vmin:.2f}, max={vmax:.2f}"
            )

    # Compare the lowest derived-T layer (1000 - 850 hPa mean, effective
    # pressure ~925 hPa, ~750 m altitude) to tas (2 m). A typical
    # tropospheric lapse rate (6.5 K/km) puts them ~5 K apart globally;
    # allow generous 15 K tolerance.
    if "tas" in ds.data_vars and "ta_derived_layer_0" in ds.data_vars:
        tas_mean = float(ds["tas"].mean())
        layer0_mean = float(ds["ta_derived_layer_0"].mean())
        diff = tas_mean - layer0_mean
        if abs(diff) > 15.0:
            messages.append(
                f"global mean tas={tas_mean:.2f} K vs ta_derived_layer_0="
                f"{layer0_mean:.2f} K differ by {diff:+.2f} K "
                "(lapse-rate sanity expects a few K)"
            )

    # Time-evolution continuity.
    if "time" in ds.dims and ds.sizes["time"] > 1:
        messages.extend(_time_continuity_messages(ds))

    return messages


def _time_delta_seconds(a, b) -> float:
    """Return ``(b - a).total_seconds()``, tolerant of both cftime
    and numpy datetime64 time coords."""
    try:
        return float((b - a).total_seconds())
    except AttributeError:
        # numpy timedelta64
        return float((b - a) / np.timedelta64(1, "s"))


_DAY_SECONDS = 86400.0
# Day-to-day jumps in global means above these magnitudes are almost
# certainly physical discontinuities (simulation boundary, restart
# with different state, corrupted write). Climatological daily swings
# in global means are <<1 for both of these under any model.
_GLOBAL_MEAN_JUMP_TOL: dict[str, float] = {
    "tas": 2.0,  # K
    "psl": 500.0,  # Pa
}


def _time_continuity_messages(ds: xr.Dataset) -> list[str]:
    """Flag non-uniform strides and unusually large day-to-day jumps
    in global means. Advisory: non-empty return goes into
    ``row.warnings`` rather than failing the dataset.
    """
    out: list[str] = []
    times = ds["time"].values
    # 1) Stride uniformity. Daily data: every gap must be 86400 s.
    strides = np.array(
        [_time_delta_seconds(times[i], times[i + 1]) for i in range(len(times) - 1)]
    )
    not_daily = np.abs(strides - _DAY_SECONDS) > 1.0  # 1 s slack
    n_bad = int(not_daily.sum())
    if n_bad:
        first_bad = int(np.argmax(not_daily))
        out.append(
            f"time stride non-uniform: {n_bad} gap(s) deviate from 86400 s "
            f"(first at index {first_bad}: {strides[first_bad]:.1f} s between "
            f"{times[first_bad]} and {times[first_bad + 1]})"
        )

    # 2) Day-to-day global-mean jumps for tas and psl. Anything above the
    # tolerances is flagged; it's very cheap (one .mean() then .diff()).
    for var, tol in _GLOBAL_MEAN_JUMP_TOL.items():
        if var not in ds.data_vars:
            continue
        gm = ds[var].mean(dim=[d for d in ds[var].dims if d != "time"]).values
        delta = np.abs(np.diff(gm))
        max_d = float(delta.max()) if delta.size else 0.0
        if max_d > tol:
            i_bad = int(np.argmax(delta))
            out.append(
                f"{var} global-mean day-to-day |delta| up to {max_d:.3g} "
                f"exceeds tol {tol:.3g} (at index {i_bad}: "
                f"{times[i_bad]} -> {times[i_bad + 1]}) — possible "
                "simulation discontinuity"
            )
    return out


def _write_zarr(ds: xr.Dataset, path: str, cfg: ResolvedDatasetConfig) -> None:
    """Write ``ds`` to ``path`` with zarr v3 chunks + shards per the
    config. time dim uses (chunk_time, shard_time); other dims are
    single chunk / single shard (full extent).
    """
    ds = _clear_stale_encoding(ds)

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
        try:
            for v in all_day_vars:
                ds_v = _open_zstore(day_zs[v])[[v]]
                ds_v, msg = _resolve_time_duplicates(ds_v, v)
                if msg:
                    row.warnings.append(msg)
                opened.append(ds_v)
        except _SimulationBoundaryError as e:
            row.status = "skipped"
            row.skip_reason = str(e)
            return row
        # join="inner" takes the intersection of coord indexes — saves
        # us from cases like ACCESS-CM2 where different day-table
        # variables for the same member are published on slightly
        # different lat/lon/time grids. Outer join (xarray's old
        # default) would expand with NaN and break downstream .dt and
        # regrid steps.
        day_ds = xr.merge(opened, compat="override", join="inner")

        # Catastrophic Pangeo-catalogue quirks can land us here with a
        # zero-length time axis — e.g. ACCESS-CM2 ssp585 r1 has some
        # variables covering 2015-2100 and others 2201-2300, so the
        # intersection is empty. Nothing we can do with that dataset;
        # skip cleanly.
        if "time" in day_ds.dims and day_ds.sizes["time"] == 0:
            row.status = "skipped"
            row.skip_reason = (
                "empty merged time dim — source variables have "
                "non-overlapping time ranges (publisher quirk)"
            )
            return row

        # 3. Cell-methods validation.
        row.cell_methods_mismatch = _validate_cell_methods(day_ds, all_day_vars)
        if row.cell_methods_mismatch:
            row.warnings.append(
                f"cell_methods != 'time: mean' for {row.cell_methods_mismatch}"
            )

        # 4. Capture native calendar.
        if "time" in day_ds.dims:
            try:
                row.native_calendar = str(day_ds["time"].dt.calendar)
            except (AttributeError, TypeError) as e:
                # Happens if the merged time coord ended up as object
                # dtype — belt-and-suspenders with the join="inner"
                # change above, which should prevent this. Log and
                # continue.
                row.warnings.append(f"could not read native calendar: {e}")

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

        # 9. Below-surface mask computed from the un-filled 3D fields.
        orog: Optional[xr.DataArray] = None
        if static_ds is not None and "orog" in static_ds:
            orog = static_ds["orog"]
        mask, row.mask_source = _compute_below_surface_mask(day_regridded, orog)

        # 10. Derive layer-mean T from the un-filled zg + hus. Running
        # this on filled zg would force dz = 0 in below-surface columns
        # (nearest-above fill copies zg[i+1] down, so zg[i] == zg[i+1])
        # and collapse the derived T to zero. Do it first, then fill.
        day_regridded = _compute_derived_layer_T(day_regridded)

        # 11. Nearest-above fill for the level-valued 3D state.
        for v in ("ua", "va", "hus", "zg"):
            if v in day_regridded:
                day_regridded[v] = _nearest_above_fill(day_regridded[v], mask)
        day_regridded = day_regridded.assign(below_surface_mask=mask)

        # 12. Cascading fill for the derived layer T — layer i is
        # invalid where either bounding level is below surface.
        day_regridded = _fill_derived_layer_T(day_regridded, mask)

        # 13. Monthly forcings -> daily. Curvilinear ocean grids (e.g.
        # SImon siconc on a tripolar grid) occasionally trip xesmf's
        # regridder with ``ESMC_FieldRegridStore failed`` or similar.
        # Catch per-forcing so the rest of the dataset still gets
        # written; the model just loses that one forcing variable.
        for table, var in (("Amon", "ts"), ("SImon", "siconc")):
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
                monthly, dup_msg = _resolve_time_duplicates(monthly, var)
                if dup_msg:
                    row.warnings.append(dup_msg)
                monthly = _apply_time_subset(monthly, cfg)
                # Keep only ``var`` as a data_var; everything else can
                # live as coords or be dropped by the regridder.
                drop = [v for v in monthly.data_vars if v != var]
                monthly = monthly.drop_vars(drop, errors="ignore")
                monthly_r, methods_used = _regrid_variables(monthly, target, cfg)
                row.regrid_methods.update(methods_used)
                interp = _interp_monthly_to_daily(
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
        sanity = _run_sanity_checks(day_regridded)
        if sanity:
            row.warnings.extend(sanity)
            for msg in sanity:
                logging.warning("  sanity: %s", msg)

        # 19. Write zarr + record variables.
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

    # Central index: merge this-run's rows with every sidecar on disk so
    # narrow --source-ids / --max-datasets re-runs don't truncate the
    # index to the subset just processed.
    all_rows = _merge_rows_for_index(rows, config.output_directory)
    write_index(all_rows, config.output_directory)

    # Final status summary covers only datasets attempted in THIS run.
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    logging.info(
        "Done. This run: %s; index now lists %d datasets.", by_status, len(all_rows)
    )


if __name__ == "__main__":
    main()
