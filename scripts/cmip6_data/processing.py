"""Shared processing utilities for CMIP6 daily data pipelines.

Data-source-agnostic helpers used by both the Pangeo pipeline
(``process.py``) and the ESGF pipeline (``process_esgf.py``):

- plev normalisation
- below-surface mask computation + nearest-above fill
- derived layer-mean T from the hypsometric equation
- xESMF regridding (with CF-bounds normalisation)
- monthly-to-daily interpolation
- time subsetting + duplicate-timestamp handling
- plev flattening into per-level 2D variables
- sanity checks
- zarr writing with chunk/shard encoding
"""

import re
from typing import Hashable, Optional

import numpy as np
import xarray as xr
from config import ResolvedDatasetConfig

# Physics constants for hypsometric layer-mean T derivation.
R_D = 287.05  # J / (kg K), dry-air gas constant
G = 9.80665  # m / s^2, standard gravity
EPS = 0.608  # (R_v / R_d) - 1, for virtual-to-actual temperature


# ---------------------------------------------------------------------------
# Duplicate-timestamp handling
# ---------------------------------------------------------------------------


class SimulationBoundaryError(ValueError):
    """Raised when duplicate time indices carry materially different
    data — a strong signal that two stitched simulations (e.g., a
    historical run and its ssp585 continuation) were concatenated into
    a single zarr without care. Silently deduplicating would hide a
    real discontinuity, so we stop and surface the issue.
    """


class DuplicateTimestampsError(ValueError):
    """Raised when duplicate time indices are detected and the
    resolved config does not enable ``allow_dedupe``. Caller is
    expected to convert this into a skipped-dataset row.
    """


def resolve_time_duplicates(
    ds: xr.Dataset, var_name: str, allow_dedupe: bool = False
) -> tuple[xr.Dataset, str]:
    """Detect duplicate timestamps and decide how to handle them.

    Returns ``(ds, message)`` where ``ds`` is the cleaned dataset and
    ``message`` is a non-empty warning string when duplicates were
    found and safely deduplicated (every duplicate pair was
    data-identical — the classic CMIP file-splice redundancy). If the
    duplicate timestamps carry *different* data, raise
    ``SimulationBoundaryError`` so the caller skips the dataset; a
    real simulation boundary needs to be split into two separate
    zarr stores, not silently merged.
    """
    if "time" not in ds.dims:
        return ds, ""
    times = ds["time"].values
    if len(times) == np.unique(times).size:
        return ds, ""

    vals, counts = np.unique(times, return_counts=True)
    dup_times = vals[counts > 1]
    n_dup = int(counts[counts > 1].sum() - len(dup_times))

    if not allow_dedupe:
        raise DuplicateTimestampsError(
            f"{var_name}: {n_dup} duplicate timestamp(s) detected; "
            "to permit dedupe, manually verify the duplicates are a "
            "publishing artefact (not a real simulation boundary) and "
            "set ``allow_dedupe: true`` in an override for this dataset"
        )

    arr = ds[var_name].load().values
    sort_idx = np.argsort(times, kind="stable")
    times_sorted = times[sort_idx]
    arr_sorted = arr[sort_idx]

    same = times_sorted[:-1] == times_sorted[1:]
    for i in np.where(same)[0]:
        a, b = arr_sorted[i], arr_sorted[i + 1]
        if not np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=0):
            raise SimulationBoundaryError(
                f"duplicate time {times_sorted[i]} in {var_name} has "
                "materially different data across copies — looks like "
                "a simulation-boundary stitch. Republish this dataset "
                "with the two halves in separate stores before "
                "ingesting."
            )

    return (
        ds.isel(time=np.unique(times, return_index=True)[1]),
        f"{var_name}: {n_dup} duplicate timestamp(s) detected "
        "(data-identical, safely deduplicated)",
    )


# ---------------------------------------------------------------------------
# Cell-methods validation
# ---------------------------------------------------------------------------

_EXPECTED_MEAN_CELL_METHODS = re.compile(r"time:\s*mean")


def validate_cell_methods(ds: xr.Dataset, variables: list[str]) -> list[str]:
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


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def grid_fingerprint(ds: xr.Dataset) -> Hashable:
    """Hashable identifier for the source dataset's lat/lon grid.

    Variables that share a fingerprint share a regridding target — we
    can merge them and build a single ``xesmf.Regridder`` per
    (grid, method). Two variables that publish on staggered grids
    (HadGEM3 u-points vs scalars, etc.) get distinct fingerprints
    and stay in separate buckets.
    """
    lat = ds["lat"].values
    lon = ds["lon"].values
    return (
        tuple(lat.shape),
        lat.tobytes(),
        tuple(lon.shape),
        lon.tobytes(),
    )


def normalize_regrid_source(ds: xr.Dataset) -> xr.Dataset:
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


def make_regridder(source_ds: xr.Dataset, target: xr.Dataset, method: str):
    """Build an xESMF regridder, importing lazily so this module can be
    loaded in envs without xESMF (e.g. unit tests on the selection logic).
    """
    import xesmf

    source_ds = normalize_regrid_source(source_ds)
    return xesmf.Regridder(source_ds, target, method, periodic=True)


def regrid_variables(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    cfg: ResolvedDatasetConfig,
) -> tuple[xr.Dataset, dict[str, str]]:
    """Regrid all data variables in ``ds`` to ``target_grid``. Returns
    the regridded dataset and a dict ``{variable: method}``.
    """
    method_for = cfg.regrid.method_for
    by_method: dict[str, list[str]] = {}
    for v in ds.data_vars:
        by_method.setdefault(method_for(v), []).append(v)

    pieces = []
    used: dict[str, str] = {}
    for method, vars_ in by_method.items():
        sub = ds[vars_]
        regridder = make_regridder(sub, target_grid, method)
        pieces.append(regridder(sub, keep_attrs=True))
        used.update({v: method for v in vars_})

    regridded = xr.merge(pieces)
    for coord in ds.coords:
        if coord not in regridded.coords and ds[coord].ndim <= 1:
            regridded = regridded.assign_coords({coord: ds[coord]})
    return regridded, used


# ---------------------------------------------------------------------------
# Plev normalisation
# ---------------------------------------------------------------------------

PLEV8_DEFAULT_HPA = np.array([1000, 850, 700, 500, 250, 100, 50, 10], dtype=np.float64)


def normalize_plev(ds: xr.Dataset) -> xr.Dataset:
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


# ---------------------------------------------------------------------------
# Below-surface mask + fill
# ---------------------------------------------------------------------------


def compute_below_surface_mask(
    ds: xr.Dataset,
    orog: Optional[xr.DataArray],
) -> tuple[Optional[xr.DataArray], str]:
    """Return (mask, source) where mask is a time-varying uint8 array
    with dims (time, plev, lat, lon). ``source`` is one of
    ``nan_union``, ``orog_static``, or ``none``.
    """
    three_d = [v for v in ("ua", "va", "hus", "zg") if v in ds.data_vars]
    if three_d:
        nan_union = ds[three_d[0]].isnull()
        for v in three_d[1:]:
            nan_union = nan_union | ds[v].isnull()
        if bool(nan_union.any()):
            return nan_union.astype("uint8").rename("below_surface_mask"), "nan_union"

    if orog is None:
        return None, "none"
    mask = (ds["zg"] < orog).astype("uint8").rename("below_surface_mask")
    return mask, "orog_static"


def nearest_above_fill(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Fill below-surface cells in ``da`` with the value at the lowest
    above-surface level in that column. Works for any number of
    consecutive masked bottom levels.

    ``da`` is (time, plev, lat, lon); ``mask`` is uint8 same shape.
    Plev axis is assumed descending in altitude (index 0 = 1000 hPa).
    """
    filled = da.where(mask == 0)
    filled = filled.bfill("plev")
    return filled


# ---------------------------------------------------------------------------
# Derived layer-mean T
# ---------------------------------------------------------------------------


def compute_derived_layer_T(ds: xr.Dataset) -> xr.Dataset:
    """Add ``ta_derived_layer(time, plev_layer, lat, lon)`` from zg + hus
    via the hypsometric equation. One layer value per gap between adjacent
    plev levels. The ``plev_layer`` coordinate stores the log-pressure
    midpoint of each layer in Pa; the dimension order matches ``plev``
    (bottom-up: index 0 = layer between the two highest-pressure levels).

    Must be called on *un-filled* zg / hus. Running this on nearest-
    above-filled zg would force ``dz = 0`` below surface and collapse
    the derived T to zero; see ``fill_derived_layer_T`` for the
    post-derivation fill.
    """
    plev_pa = ds["plev"].values
    z = ds["zg"]
    q = ds["hus"]
    layers = []
    n_layers = len(plev_pa) - 1
    for i in range(n_layers):
        p_lo = plev_pa[i]
        p_hi = plev_pa[i + 1]
        dz = z.isel(plev=i + 1) - z.isel(plev=i)
        q_mean = 0.5 * (q.isel(plev=i) + q.isel(plev=i + 1))
        tv = G * dz / (R_D * np.log(p_lo / p_hi))
        t = tv / (1.0 + EPS * q_mean)
        layers.append(t.drop_vars("plev", errors="ignore"))
    plev_layer = np.array(
        [np.sqrt(plev_pa[i] * plev_pa[i + 1]) for i in range(n_layers)]
    )
    ta = xr.concat(layers, dim="plev_layer").assign_coords(plev_layer=plev_layer)
    ta.attrs = {
        "long_name": "derived layer-mean temperature from hypsometric equation",
        "units": "K",
        "derivation": "hypsometric from zg + hus",
    }
    return ds.assign(ta_derived_layer=ta)


def fill_derived_layer_T(ds: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """Cascading nearest-above fill for ``ta_derived_layer``.

    Layer ``i`` (in the internal bottom-up ordering along ``plev_layer``)
    is treated as invalid where either bounding plev level (``plev[i]``
    or ``plev[i+1]``) is below-surface per ``mask``, since the
    hypsometric formula uses the model's below-surface extrapolation
    there and produces unphysical values.

    The fill cascades top-down: the topmost layer is always valid
    (stratosphere); each layer below inherits the layer above where
    invalid.
    """
    if "ta_derived_layer" not in ds.data_vars:
        return ds
    ta = ds["ta_derived_layer"]
    n_layers = ta.sizes["plev_layer"]
    if n_layers < 2:
        return ds
    for i in range(n_layers - 2, -1, -1):
        layer_mask = (mask.isel(plev=i) | mask.isel(plev=i + 1)).astype(bool)
        ta_i = ta.isel(plev_layer=i)
        ta_above = ta.isel(plev_layer=i + 1)
        ta.loc[dict(plev_layer=ta["plev_layer"].values[i])] = ta_i.where(
            ~layer_mask, ta_above
        )
    ds["ta_derived_layer"] = ta
    return ds


# ---------------------------------------------------------------------------
# Monthly -> daily interpolation
# ---------------------------------------------------------------------------


def interp_monthly_to_daily(
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
    return interp.bfill("time").ffill("time")


# ---------------------------------------------------------------------------
# Time subsetting
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def clip_date_for_calendar(date_str: str, calendar: str) -> str:
    """Clip a ``YYYY-MM-DD`` string to a valid day in the target
    calendar. Mostly matters for ``360_day``, where every month has
    exactly 30 days.
    """
    if calendar != "360_day":
        return date_str
    m = _ISO_DATE_RE.match(date_str)
    if not m:
        return date_str
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    d = min(d, 30)
    return f"{y:04d}-{mo:02d}-{d:02d}"


def apply_time_subset(ds: xr.Dataset, cfg: ResolvedDatasetConfig) -> xr.Dataset:
    """Restrict ``ds`` to ``cfg.time_subset[cfg.experiment]``.

    Uses a boolean mask + ``isel`` rather than ``ds.sel(time=slice(...))``
    so a source with duplicate time indices doesn't blow up with
    ``Cannot get left slice bound for non-unique label``.
    """
    window = cfg.time_subset.get(cfg.experiment)
    if window is None or "time" not in ds.dims:
        return ds

    try:
        calendar = str(ds["time"].dt.calendar)
    except (AttributeError, TypeError):
        calendar = "standard"
    start = clip_date_for_calendar(window.start, calendar)
    end = clip_date_for_calendar(window.end, calendar)

    times = ds["time"].values
    if len(times) and hasattr(times[0], "calendar"):
        date_type = type(times[0])
        sy, sm, sd = (int(x) for x in start.split("-"))
        ey, em, ed = (int(x) for x in end.split("-"))
        start_dt = date_type(sy, sm, sd)
        end_dt = date_type(ey, em, ed)
    else:
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end)
    mask = (times >= start_dt) & (times <= end_dt)
    return ds.isel(time=np.where(mask)[0])


# ---------------------------------------------------------------------------
# Encoding + zarr write
# ---------------------------------------------------------------------------

_STALE_ENCODING_KEYS = (
    "compressors",
    "filters",
    "preferred_chunks",
    "chunks",
    "shards",
    "_FillValue",
    "missing_value",
    "dtype",
)


def clear_stale_encoding(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    for var in {**ds.coords, **ds.data_vars}.values():
        for k in _STALE_ENCODING_KEYS:
            var.encoding.pop(k, None)
    return ds


def write_zarr(ds: xr.Dataset, path: str, cfg: ResolvedDatasetConfig) -> None:
    """Write ``ds`` to ``path`` with zarr v3 chunks + shards per the
    config. time dim uses (chunk_time, shard_time); other dims are
    single chunk / single shard (full extent).
    """
    ds = clear_stale_encoding(ds)

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


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

_EPS = 0.01

_SANITY_RANGES: dict[str, tuple[float, float]] = {
    "ua": (-200.0, 200.0),
    "va": (-200.0, 200.0),
    "uas": (-100.0, 100.0),
    "vas": (-100.0, 100.0),
    "sfcWind": (-_EPS, 100.0),
    "hus": (-_EPS, 0.05),
    "huss": (-_EPS, 0.05),
    "ts": (180.0, 340.0),
    "tas": (180.0, 340.0),
    "psl": (8.0e4, 1.1e5),
    "pr": (-_EPS, 0.01),
    "siconc": (-_EPS, 100.0 + _EPS),
    "rsdt": (-_EPS, 600.0),
    "rsut": (-_EPS, 600.0),
    "rlut": (-_EPS, 400.0),
    "rsds": (-_EPS, 600.0),
    "rsus": (-_EPS, 600.0),
    "rlds": (-_EPS, 600.0),
    "rlus": (-_EPS, 700.0),
    "hfss": (-1000.0, 1000.0),
    "hfls": (-500.0, 1200.0),
    "sftlf": (-_EPS, 105.0),
    "orog": (-500.0, 9000.0),
}

_DERIVED_T_RANGE = (150.0, 350.0)

_DAY_SECONDS = 86400.0
_GLOBAL_MEAN_JUMP_TOL: dict[str, float] = {
    "tas": 2.0,
    "psl": 500.0,
}


def _time_delta_seconds(a, b) -> float:
    try:
        return float((b - a).total_seconds())
    except AttributeError:
        return float((b - a) / np.timedelta64(1, "s"))


def _time_continuity_messages(ds: xr.Dataset) -> list[str]:
    out: list[str] = []
    times = ds["time"].values
    strides = np.array(
        [_time_delta_seconds(times[i], times[i + 1]) for i in range(len(times) - 1)]
    )
    not_daily = np.abs(strides - _DAY_SECONDS) > 1.0
    n_bad = int(not_daily.sum())
    if n_bad:
        first_bad = int(np.argmax(not_daily))
        out.append(
            f"time stride non-uniform: {n_bad} gap(s) deviate from 86400 s "
            f"(first at index {first_bad}: {strides[first_bad]:.1f} s between "
            f"{times[first_bad]} and {times[first_bad + 1]})"
        )

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


def run_sanity_checks(ds: xr.Dataset) -> list[str]:
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

    if "ta_derived_layer" in ds.data_vars:
        lo, hi = _DERIVED_T_RANGE
        arr = ds["ta_derived_layer"]
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmin < lo or vmax > hi:
            messages.append(
                f"ta_derived_layer out of range [{lo}, {hi}] K: "
                f"min={vmin:.2f}, max={vmax:.2f}"
            )

    if "tas" in ds.data_vars and "ta_derived_layer" in ds.data_vars:
        tas_mean = float(ds["tas"].mean())
        layer0_mean = float(ds["ta_derived_layer"].isel(plev_layer=0).mean())
        diff = tas_mean - layer0_mean
        if abs(diff) > 15.0:
            messages.append(
                f"global mean tas={tas_mean:.2f} K vs ta_derived_layer "
                f"lowest layer={layer0_mean:.2f} K differ by {diff:+.2f} K "
                "(lapse-rate sanity expects a few K)"
            )

    if "time" in ds.dims and ds.sizes["time"] > 1:
        messages.extend(_time_continuity_messages(ds))

    return messages


# ---------------------------------------------------------------------------
# Plev flattening
# ---------------------------------------------------------------------------


def plev_hpa_label(plev_pa: float) -> str:
    """Format a pressure value in Pa as an integer hPa string."""
    return str(round(plev_pa / 100))


def flatten_plev_variables(ds: xr.Dataset) -> xr.Dataset:
    """Split variables with a ``plev`` or ``plev_layer`` dimension into
    per-level 2D variables named by pressure.

    On-level variables (``plev`` dim) are named ``{var}{hPa}``, e.g.
    ``ua1000``, ``ua850``, ..., ``ua10``.

    Between-level derived layers (``plev_layer`` dim) are named
    ``{var}_{lo_hPa}_{hi_hPa}``, e.g. ``ta_derived_layer_1000_850``.

    Both ``plev`` and ``plev_layer`` coordinates are dropped from the
    returned dataset, making all variables uniformly
    ``(time, lat, lon)`` or ``(lat, lon)``.
    """
    plev_vars = [v for v in ds.data_vars if "plev" in ds[v].dims]
    layer_vars = [v for v in ds.data_vars if "plev_layer" in ds[v].dims]
    if not plev_vars and not layer_vars:
        return ds

    new_vars: dict[str, xr.DataArray] = {}

    if plev_vars:
        plev_pa = ds["plev"].values
        for v in plev_vars:
            da = ds[v]
            for i in range(da.sizes["plev"]):
                label = plev_hpa_label(plev_pa[i])
                level_da = da.isel(plev=i).drop_vars("plev", errors="ignore")
                new_vars[f"{v}{label}"] = level_da

    if layer_vars:
        plev_pa = ds["plev"].values
        for v in layer_vars:
            da = ds[v]
            for i in range(da.sizes["plev_layer"]):
                lo = plev_hpa_label(plev_pa[i])
                hi = plev_hpa_label(plev_pa[i + 1])
                level_da = da.isel(plev_layer=i).drop_vars(
                    "plev_layer", errors="ignore"
                )
                new_vars[f"{v}_{lo}_{hi}"] = level_da

    ds = ds.drop_vars(plev_vars + layer_vars)
    ds = ds.drop_vars(["plev", "plev_layer"], errors="ignore")
    return ds.assign(**new_vars)


__all__ = [
    "SimulationBoundaryError",
    "DuplicateTimestampsError",
    "resolve_time_duplicates",
    "validate_cell_methods",
    "grid_fingerprint",
    "normalize_regrid_source",
    "make_regridder",
    "regrid_variables",
    "PLEV8_DEFAULT_HPA",
    "normalize_plev",
    "compute_below_surface_mask",
    "nearest_above_fill",
    "compute_derived_layer_T",
    "fill_derived_layer_T",
    "interp_monthly_to_daily",
    "clip_date_for_calendar",
    "apply_time_subset",
    "clear_stale_encoding",
    "write_zarr",
    "run_sanity_checks",
    "plev_hpa_label",
    "flatten_plev_variables",
]
