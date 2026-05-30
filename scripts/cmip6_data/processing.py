"""Shared processing utilities for CMIP6 daily data pipelines.

Data-source-agnostic helpers used by both the Pangeo pipeline
(``process.py``) and the ESGF pipeline (``process_esgf.py``):

- plev normalisation
- below-surface mask computation + nearest-above fill
- xESMF regridding (with CF-bounds normalisation)
- causal monthly-to-daily mapping for surface-and-ocean variables
- time subsetting + duplicate-timestamp handling
- plev flattening into per-level 2D variables
- sanity checks
- zarr writing with chunk/shard encoding
"""

import logging
import re
import threading
import time
from typing import Hashable, Optional

import numpy as np
import xarray as xr
from config import ResolvedDatasetConfig, SurfaceAndOceanVariable

BOUNDS_NAMES: frozenset[str] = frozenset(
    {
        "lon_bnds",
        "lat_bnds",
        "lon_b",
        "lat_b",
        "vertices_longitude",
        "vertices_latitude",
    }
)


def rss_mib() -> float:
    """Resident-set memory in MiB. Returns NaN if psutil is unavailable
    so callers can log unconditionally without try/except."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return float("nan")


class RssSampler:
    """Daemon thread that logs RSS at a fixed interval. Use ``start()``
    / ``stop()`` around the per-task body, or as a context manager.
    Catches slow drift between stage boundaries that the per-stage RSS
    log alone would miss.
    """

    def __init__(self, interval_seconds: float = 300.0) -> None:
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "RssSampler":
        t0 = time.monotonic()

        def _loop() -> None:
            while not self._stop_event.wait(self._interval):
                logging.info(
                    "  [rss sampler] +%.0fs rss=%.0f MiB",
                    time.monotonic() - t0,
                    rss_mib(),
                )

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "RssSampler":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


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

    # Only load the duplicate-timestamp slices, not the whole variable.
    # On models like CESM2-WACCM, the full 3D variable for the configured
    # window is ~50+ GB at native resolution; loading it just to compare a
    # few hundred duplicate pairs OOMs the pod. Per-duplicate-timestamp
    # load is bounded by (n_copies × per-timestep) which is ~MB-scale.
    da = ds[var_name]
    for t_val in dup_times:
        positions = np.where(times == t_val)[0].tolist()
        copies = da.isel(time=positions).load().values
        ref = copies[0]
        for c in copies[1:]:
            if not np.allclose(ref, c, equal_nan=True, rtol=1e-5, atol=0):
                raise SimulationBoundaryError(
                    f"duplicate time {t_val} in {var_name} has "
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
        # CMIP6 curvilinear ocean files (Oday/Omon on tripolar grids
        # like ACCESS-ESM1-5, EC-Earth3, etc.) carry 2D coords named
        # ``longitude``/``latitude`` on dimensions like ``(j, i)``.
        # xesmf only looks up ``lon``/``lat`` — without this rename,
        # ``xesmf.Regridder`` raises "dataset must include lon/lat or
        # be CF-compliant" even though the data has perfectly valid
        # coordinates. CMIP6 rectilinear atmos files name them
        # ``lon``/``lat`` already so this is a no-op there.
        ("longitude", "lon"),
        ("latitude", "lat"),
    ):
        if src in ds.variables and dst not in ds.variables:
            ds = ds.rename({src: dst})

    for name in ("lon_b", "lat_b"):
        if name not in ds.variables:
            continue
        # Load the bounds DataArray before ``bounds_to_vertices`` — that
        # helper calls ``apply_ufunc`` with the 4-vertex dim as a core
        # dim and refuses to run when the dim is chunked. CMIP6 ocean
        # zarrs (Oday.tos, Omon.* on curvilinear grids) ship with
        # ``vertices`` pre-chunked, which broke ~150 datasets in the
        # pilot. The bounds array itself is tiny (lat × lon × 4 ×
        # 8 bytes ≈ a few MB at native resolution), so eagerly loading
        # it is cheap.
        da = ds[name].load()
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


def _has_bounds(ds: xr.Dataset) -> bool:
    return "lon_b" in ds.variables and "lat_b" in ds.variables


def is_unstructured_source(ds: xr.Dataset) -> bool:
    """A source is unstructured when ``lat`` and ``lon`` are both 1D and
    share a non-canonical dim (e.g. ``ncells`` on AWI's FESOM ocean
    grid).
    """
    if "lat" not in ds.variables or "lon" not in ds.variables:
        return False
    lat = ds["lat"]
    lon = ds["lon"]
    if lat.ndim != 1 or lon.ndim != 1 or lat.dims != lon.dims:
        return False
    return lat.dims[0] not in ("lat", "lon")


# Sentinel method string returned by ``make_regridder`` for unstructured
# sources. Distinct from plain ``nearest_s2d`` so callers can detect the
# unstructured path and apply a target-grid land mask (xesmf with
# ``locstream_in=True`` has no land cells in the source and produces
# valid values everywhere, including over land).
UNSTRUCTURED_METHOD = "nearest_s2d_locstream"


def make_regridder(source_ds: xr.Dataset, target: xr.Dataset, method: str) -> tuple:
    """Build an xESMF regridder, importing lazily so this module can be
    loaded in envs without xESMF (e.g. unit tests on the selection logic).

    Returns ``(regridder, actual_method)`` — the method may differ from the
    request if conservative was requested but no grid bounds are available
    (common for ocean-grid variables like siconc), or if the source is
    unstructured (AWI FESOM), in which case xesmf only supports
    ``nearest_s2d`` via ``locstream_in=True``.
    """
    import xesmf

    source_ds = normalize_regrid_source(source_ds)

    if is_unstructured_source(source_ds):
        return (
            xesmf.Regridder(source_ds, target, "nearest_s2d", locstream_in=True),
            UNSTRUCTURED_METHOD,
        )

    actual_method = method
    if method == "conservative" and not _has_bounds(source_ds):
        logging.warning(
            "  no lon_b/lat_b for conservative regridding, falling back to bilinear"
        )
        actual_method = "bilinear"
    return (
        xesmf.Regridder(source_ds, target, actual_method, periodic=True),
        actual_method,
    )


def regrid_variables(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    cfg: ResolvedDatasetConfig,
) -> tuple[xr.Dataset, dict[str, str]]:
    """Regrid all data variables in ``ds`` to ``target_grid``. Returns
    the regridded dataset and a dict ``{variable: method}``.
    """
    method_for = cfg.regrid.method_for
    bounds_vars = [v for v in ds.data_vars if v in BOUNDS_NAMES]
    by_method: dict[str, list[str]] = {}
    for v in ds.data_vars:
        if v not in BOUNDS_NAMES:
            by_method.setdefault(method_for(v), []).append(v)

    pieces = []
    used: dict[str, str] = {}
    for method, vars_ in by_method.items():
        sub = ds[vars_ + bounds_vars]
        regridder, actual_method = make_regridder(sub, target_grid, method)
        # Apply per-variable instead of as a batched ``regridder(ds[vars_])``
        # — the compute cost is identical (the sparse matmul is per-array
        # inside xesmf either way) but per-variable lets dask release each
        # variable's source-resolution chunk before allocating the next.
        # Peak memory drops from Σ(vars in bucket) × per-chunk to
        # max(vars in bucket) × per-chunk, which matters a lot for
        # high-resolution sources like HadGEM3-GC31-MM (~6.5 GB per 3D
        # plev variable per chunk).
        #
        # ``skipna=True`` makes the conservative / bilinear weighted
        # average skip source NaN cells instead of treating them as 0.
        # Without it, a target cell that partially covers source NaN
        # (e.g. sitemptop where ice is only in a small sub-cell patch,
        # or oday_tos near a coastline that crosses both land=NaN and
        # ocean) gets a value pulled toward 0 by the NaN cells'
        # implicit-zero contribution — for sitemptop in the pilot, that
        # meant cohort means of ~88 K under the ice mask instead of the
        # physically reasonable ~268 K. xesmf's default ``na_thres=1.0``
        # still propagates NaN to target cells whose entire source
        # footprint was NaN, so the post-regrid mask detection in
        # ``emit_mask_and_fill`` still works.
        for v in vars_:
            pieces.append(regridder(ds[[v]], keep_attrs=True, skipna=True))
        used.update({v: actual_method for v in vars_})

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

    # The ``orog_static`` fallback needs ``zg`` to compare against the
    # static orography. Models that publish only a subset of the 3D
    # state (e.g. CMCC-CM2-SR5, which ships only ``ua`` of the four
    # core 3D variables) get here without ``zg``; in that case there's
    # no information to build a mask from, so return ``"none"``.
    if orog is None or "zg" not in ds.data_vars:
        return None, "none"
    mask = (ds["zg"] < orog).astype("uint8").rename("below_surface_mask")
    return mask, "orog_static"


def nearest_above_fill(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Legacy below-surface fill: each column's masked cells inherit the
    lowest above-surface level's value.

    Kept in the module for the 0.2.0→0.3.0 migration regression test —
    it simulates how older schema versions filled below-surface cells
    before the switch to :func:`fill_below_surface_smooth`. The
    production pipeline no longer calls this function; use
    :func:`fill_below_surface_smooth` for new data.

    ``da`` is (time, plev, lat, lon); ``mask`` is uint8 same shape.
    Plev axis is assumed descending in altitude (index 0 = 1000 hPa).
    """
    filled = da.where(mask == 0)
    filled = filled.bfill("plev")
    return filled


def fill_below_surface_smooth(
    da: xr.DataArray,
    mask: xr.DataArray,
    num_steps: int = 4,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0,
) -> xr.DataArray:
    """Fill below-surface cells in ``da`` using a smooth flood fill —
    the same algorithm the model applies at runtime in
    ``fme.core.fill``. The implementation here is a numpy/scipy port
    (see :mod:`fill`); the dev parity test ``test_fill.py`` keeps it
    bit-comparable to the torch reference so the ingest container can
    stay torch-free.

    For each plev level we set the masked cells to NaN, precompute a
    static (lat, lon) interior mask from the time-union of NaN cells,
    and dispatch :func:`fill.fast_flood_fill` with that interior mask.
    Per-(time) NaN patterns drive the iterative edge-blend loop, so
    cells that are sometimes-below-surface still keep their real data
    on the timesteps when they're above surface.

    ``da`` is (time, plev, lat, lon); ``mask`` is uint8 same shape with
    1 marking below-surface cells.
    """
    from fill import fast_flood_fill, get_interior_mask

    dims = ("time", "plev", "lat", "lon")
    if da.dims != dims or mask.dims != dims:
        raise ValueError(
            f"fill_below_surface_smooth expects dims {dims}, got "
            f"da={da.dims} mask={mask.dims}"
        )

    out = da.copy().astype(np.float32)
    n_plev = da.sizes["plev"]
    H, W = da.sizes["lat"], da.sizes["lon"]
    for p in range(n_plev):
        plane = out.isel(plev=p).values  # (time, lat, lon)
        mask_plane = mask.isel(plev=p).values  # (time, lat, lon)
        if not mask_plane.any():
            continue
        # NaN the below-surface cells per timestep.
        plane = np.where(mask_plane.astype(bool), np.nan, plane).astype(np.float32)
        # Union-of-time interior mask, computed once for this level.
        union_nan = mask_plane.any(axis=0)
        if not union_nan.any():
            out.values[:, p] = plane  # type: ignore[index]
            continue
        interior = get_interior_mask(union_nan, num_steps=num_steps).reshape(1, 1, H, W)
        filled = fast_flood_fill(
            plane[None],  # (1, T, H, W)
            num_steps=num_steps,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            interior_mask=interior,
        )
        out.values[:, p] = filled[0]  # type: ignore[index]
    return out


def fill_horizontal_diffuse(
    da: xr.DataArray,
    max_iterations: int = 50,
) -> xr.DataArray:
    """Fill NaN cells in a 2D (lat, lon) or 3D (time, lat, lon) array via
    iterative nearest-neighbor diffusion.

    At each iteration, every NaN cell that has at least one finite
    neighbor (4-connected: N, S, E, W with longitude wraparound) takes
    the mean of those finite neighbors. After ``max_iterations``, any
    cell still NaN (entire array NaN at that timestep) is filled with
    the timestep's mean of originally-finite cells, or 0 if no finite
    cells remained.

    Unlike ``nearest_above_fill`` (column-wise, per plev), this works
    for arbitrary 2D NaN shapes — e.g., continental land masses for
    ocean variables, or land + ice-free ocean for sea-ice variables.
    Boundary in latitude clamps at the poles (no wrap); longitude
    wraps periodically.

    Returns a new DataArray; the input is not modified.
    """
    from scipy.ndimage import convolve

    arr = np.asarray(da.values, dtype=np.float64)
    if arr.ndim == 2:
        out = _diffuse_one_plane(arr, max_iterations, convolve)
        return da.copy(data=out.astype(da.dtype))
    if arr.ndim != 3:
        raise ValueError(
            f"fill_horizontal_diffuse expects 2D or 3D array, got {arr.ndim}D"
        )
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        out[t] = _diffuse_one_plane(arr[t], max_iterations, convolve)
    return da.copy(data=out.astype(da.dtype))


def _diffuse_one_plane(plane: np.ndarray, max_iterations: int, convolve) -> np.ndarray:
    """One (lat, lon) timestep of the diffusion fill. ``plane`` may
    contain NaN; returns a fully-finite array.
    """
    valid = np.isfinite(plane)
    if not valid.any():
        return np.zeros_like(plane)
    values = np.where(valid, plane, 0.0)
    # 4-connected kernel — no diagonals, keeps the diffusion isotropic
    # along lat/lon axes.
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for _ in range(max_iterations):
        if valid.all():
            break
        # Periodic in lon (wrap), clamp in lat (nearest). scipy.ndimage
        # uses a single mode per call; we manually wrap longitude by
        # padding and convolving with mode='constant'.
        values_pad = np.concatenate([values[:, -1:], values, values[:, :1]], axis=1)
        valid_pad = np.concatenate([valid[:, -1:], valid, valid[:, :1]], axis=1)
        sum_n = convolve(values_pad, kernel, mode="nearest")[:, 1:-1]
        cnt_n = convolve(valid_pad.astype(np.float64), kernel, mode="nearest")[:, 1:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            new_v = np.where(cnt_n > 0, sum_n / np.maximum(cnt_n, 1e-30), 0.0)
        newly = (~valid) & (cnt_n > 0)
        values = np.where(newly, new_v, values)
        valid = valid | newly
    if not valid.all():
        # Any remaining NaN island (disconnected from valid cells) gets
        # the plane mean of originally-finite cells.
        original_valid = np.isfinite(plane)
        fallback = float(plane[original_valid].mean()) if original_valid.any() else 0.0
        values = np.where(valid, values, fallback)
    return values


# ---------------------------------------------------------------------------
# Monthly -> daily interpolation
# ---------------------------------------------------------------------------


def causal_annual_to_daily(
    annual: xr.DataArray,
    daily_time: xr.DataArray,
) -> xr.DataArray:
    """Map annual values onto a daily time axis using the previous
    calendar year's value — strictly causal, no future leakage.
    Piecewise constant: every day in calendar year ``Y`` takes the
    same value, year ``Y-1``'s mean.

    Same start-of-series fallback as :func:`causal_monthly_to_daily`:
    when the previous year isn't in the annual series, the first
    available annual value is used as a constant-extrapolation.
    """
    annual_years = np.array([int(t.year) for t in annual["time"].values])
    daily_years = np.array([int(t.year) for t in daily_time.values])
    target_years = daily_years - 1

    year_to_idx: dict[int, int] = {int(y): i for i, y in enumerate(annual_years)}
    first_idx = 0
    day_idx = np.array([year_to_idx.get(int(y), first_idx) for y in target_years])
    out = annual.isel(time=day_idx)
    return out.assign_coords(time=daily_time.values)


def causal_monthly_to_daily(
    monthly: xr.DataArray,
    daily_time: xr.DataArray,
) -> xr.DataArray:
    """Map monthly values onto a daily time axis using the previous
    calendar month's value — strictly causal, no future leakage.
    Piecewise constant: every day in calendar month ``M`` takes the
    same value, month ``M-1``'s mean.

    For each daily timestamp at calendar month ``M``, the returned
    value is the monthly mean for month ``M-1``. At the start of the
    series, when the previous month is not available, the first
    monthly value is used as a constant-extrapolation fallback.

    Works for any calendar (standard, noleap, 360_day, etc.) by keying
    on the ``(year, month)`` tuple of each timestamp.
    """
    monthly_ym = np.array([(int(t.year), int(t.month)) for t in monthly["time"].values])
    daily_ym = np.array([(int(t.year), int(t.month)) for t in daily_time.values])

    target_ym = daily_ym.copy()
    target_ym[:, 1] -= 1
    wrap = target_ym[:, 1] == 0
    target_ym[wrap, 0] -= 1
    target_ym[wrap, 1] = 12

    month_to_idx: dict[tuple[int, int], int] = {
        (int(y), int(m)): i for i, (y, m) in enumerate(monthly_ym)
    }
    # Constant-extrapolation fallback: any target month before the
    # earliest monthly entry maps to the first entry.
    first_idx = 0
    day_idx = np.array(
        [month_to_idx.get((int(y), int(m)), first_idx) for y, m in target_ym]
    )

    out = monthly.isel(time=day_idx)
    return out.assign_coords(time=daily_time.values).rename({"time": "time"})


# CMIP6 publishes some temperature variables in °C (``tos``, ``tob``,
# ``sitemptop``) and others in K (``ts``, ``tas``, ``ta``), but
# publishers occasionally deviate from the CMOR default. Force K
# everywhere by inspecting the units attribute and converting when
# it indicates Celsius. ``tossq`` is the square of SST — units are
# variance and a linear offset doesn't apply; left alone with a
# warning when detected.
_CELSIUS_UNIT_TOKENS: frozenset[str] = frozenset(
    {
        "degc",
        "degrees_c",
        "degreesc",
        "degrees c",
        "deg_c",
        "deg c",
        "celsius",
        "degcelsius",
        "deg celsius",
        "°c",
        "c",
    }
)
_KELVIN_UNIT_TOKENS: frozenset[str] = frozenset({"k", "kelvin", "deg_k", "degk"})
_TEMPERATURE_VARS_SPEC_CELSIUS: frozenset[str] = frozenset({"tos", "tob", "sitemptop"})
_TEMPERATURE_VARS_SPEC_KELVIN: frozenset[str] = frozenset({"tas", "ts", "ta"})


def harmonize_temperature_to_kelvin(
    da: xr.DataArray, var_id: str = ""
) -> tuple[xr.DataArray, str]:
    """Return ``(da_in_kelvin, message)`` for a temperature variable.

    If ``da.attrs["units"]`` reads as Celsius (any of the common
    spellings), add 273.15 and set ``units = "K"``. If the units
    attribute is missing, fall back to the CMIP6 CMOR spec default
    for the given ``var_id``. Already-K variables are returned
    unchanged. ``message`` is non-empty when a conversion happened or
    a suspect attribute was detected.

    Preserves all other attrs across the unit conversion.
    """
    raw_units = str(da.attrs.get("units", "")).strip()
    units_token = raw_units.lower().replace("**", "").replace("^", "")
    if not units_token:
        if var_id in _TEMPERATURE_VARS_SPEC_CELSIUS:
            units_token = "degc"
        elif var_id in _TEMPERATURE_VARS_SPEC_KELVIN:
            units_token = "k"
    if units_token in _CELSIUS_UNIT_TOKENS:
        attrs = dict(da.attrs)
        converted = da + 273.15
        converted.attrs = attrs
        converted.attrs["units"] = "K"
        return converted, (
            f"{var_id or da.name}: converted °C → K"
            f" (source attr units={raw_units!r})"
        )
    if units_token in _KELVIN_UNIT_TOKENS or not units_token:
        # Already K (or missing both attr and a spec default).
        return da, ""
    # Units don't look like temperature at all (e.g. ``W m-2``, ``Pa``,
    # ``kg m-2 s-1``). Process-side this helper is called once per
    # variable in the output dataset, so silently ignore non-temperature
    # variables. Only warn when we have a strong prior that ``var_id``
    # should be a temperature (its bare CMIP6 name is in the spec
    # lists, or the rename map points it at a known temperature output
    # name) but the units don't match.
    looks_temperature = (
        var_id in _TEMPERATURE_VARS_SPEC_CELSIUS
        or var_id in _TEMPERATURE_VARS_SPEC_KELVIN
        or var_id.startswith("TMP")
        or var_id.endswith(("_ts", "_tos", "_tob", "_sitemptop"))
    )
    if not looks_temperature:
        return da, ""
    return da, (
        f"{var_id or da.name}: unrecognized temperature units "
        f"{raw_units!r}, left unconverted"
    )


def apply_output_renames(ds: xr.Dataset, rename_map: dict[str, str]) -> xr.Dataset:
    """Rename variables according to ``rename_map`` and tag each renamed
    variable with an ``original_name`` attribute pointing to its bare
    CMIP6 source name.

    Variables not in ``rename_map`` are left alone. Variables in
    ``rename_map`` that aren't in ``ds`` are silently skipped (the
    map covers all radiative-flux renames; many models don't publish
    every variable).
    """
    applicable = {src: dst for src, dst in rename_map.items() if src in ds.data_vars}
    if not applicable:
        return ds
    out = ds.rename(applicable)
    for src, dst in applicable.items():
        out[dst].attrs["original_name"] = src
    return out


def compute_total_water_path(
    water_vapor_path: xr.DataArray,
    cloud_condensed_water_path: xr.DataArray,
) -> xr.DataArray:
    """Total atmospheric water column = vapor + condensed cloud
    (liquid + ice). Units kg m⁻².

    The CM4/SHIELD baseline emits a single ``total_water_path`` field
    that includes both vapor and cloud water. CMIP6 splits these into
    ``prw`` (vapor) and ``clwvi`` (condensed water path). We emit both
    inputs *and* the derived sum, so consumers can pick whichever
    representation matches what their training data uses.
    """
    total = water_vapor_path + cloud_condensed_water_path
    out = total.rename("total_water_path")
    out.attrs["original_name"] = "derived"
    out.attrs["long_name"] = (
        "total water path (vapor + cloud condensate) " "= water_vapor_path + clwvi"
    )
    out.attrs["units"] = "kg m-2"
    return out


def derive_ocean_and_correct_sea_ice(
    land_fraction: xr.DataArray,
    sea_ice_fraction: xr.DataArray,
    ocean_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Derive ``ocean_fraction`` and correct ``sea_ice_fraction`` so the
    identity ``land + ice + ocean = 1`` holds exactly in every cell.

    Where ``land + ice > 1`` (typically coastal cells, or cells where
    sea-ice was diffused over the land mask by ``emit_mask_and_fill``),
    the excess is moved back into ``sea_ice_fraction`` — matching the
    convention used by the ERA5 build pipeline. Returns the corrected
    sea-ice and the derived ocean, both with ``(time, lat, lon)`` dim
    order to match what the zarr writer expects.

    ``land_fraction`` is static; ``sea_ice_fraction`` is time-varying.
    xarray's broadcasting handles the shape; we explicitly transpose
    the outputs so the time axis comes first.
    """
    raw_ocean = 1.0 - land_fraction - sea_ice_fraction
    # Negative ocean means land+ice > 1 in that cell — push the excess
    # back into ``sea_ice_fraction`` and clip ocean to zero there.
    negative_ocean = raw_ocean.where(raw_ocean < 0, 0.0)
    ocean = (raw_ocean - negative_ocean).clip(0.0, 1.0)
    corrected_ice = (sea_ice_fraction + negative_ocean).clip(0.0, 1.0)

    ocean = ocean.transpose(*sea_ice_fraction.dims).rename(ocean_name)
    ocean.attrs["original_name"] = "derived"
    ocean.attrs["long_name"] = (
        "ocean fraction = 1 - land_fraction - sea_ice_fraction "
        "(budget-corrected so land+ice+ocean=1)"
    )
    ocean.attrs["units"] = "1"

    corrected_ice = corrected_ice.transpose(*sea_ice_fraction.dims).rename(
        sea_ice_fraction.name
    )
    corrected_ice.attrs = dict(sea_ice_fraction.attrs)
    return corrected_ice, ocean


# Backwards-compatible wrapper used by tests written against the
# original single-return signature. Prefer the corrected pair via
# ``derive_ocean_and_correct_sea_ice`` from new call sites.
def compute_ocean_fraction(
    land_fraction: xr.DataArray,
    sea_ice_fraction: xr.DataArray,
    name: str,
) -> xr.DataArray:
    _, ocean = derive_ocean_and_correct_sea_ice(land_fraction, sea_ice_fraction, name)
    return ocean


def apply_target_land_mask(
    da: xr.DataArray, land_fraction: xr.DataArray, threshold: float = 0.5
) -> xr.DataArray:
    """NaN-fill cells where ``land_fraction > threshold`` (mostly land).

    ``land_fraction`` is on [0, 1] (post ``clamp_static_fractions``
    rescale + rename of ``sftlf``); the default ``threshold=0.5``
    matches the pre-rescale ``sftlf > 50`` convention.

    For ocean/sea-ice variables whose source grid has no land cells
    (FESOM and other unstructured ocean grids), the regridded output
    holds a valid value in every target cell — the nearest source
    point is always an ocean point. Applying the target-grid land
    mask restores the NaN-over-land pattern downstream consumers
    expect, including ``emit_mask_and_fill``.
    """
    return da.where(land_fraction <= threshold)


def finalize_surface_and_ocean_variable(
    regridded: xr.DataArray,
    var: SurfaceAndOceanVariable,
    daily_time: xr.DataArray,
    fill_iterations: int = 50,
) -> dict[str, xr.DataArray]:
    """Given a regridded source variable, produce the named output(s)
    for the dataset: filled values under ``var.output_name``, plus a
    ``{output_name}_mask`` channel for ocean / sea-ice kinds.

    Applies cadence-specific time mapping:
    - ``daily`` cadence: reindex source to ``daily_time`` via
      nearest-neighbor (source timestamps may differ from the daily
      axis by a few hours, e.g. day-center vs day-start).
    - ``monthly_causal`` cadence: map each day to the previous calendar
      month's value via :func:`causal_monthly_to_daily`.

    Returns a dict ``{output_name: DataArray}`` ready to assign into
    the output dataset.
    """
    if var.cadence == "monthly_causal":
        on_daily = causal_monthly_to_daily(regridded, daily_time)
    elif var.cadence == "daily":
        on_daily = regridded.reindex(time=daily_time.values, method="nearest")
    else:
        raise ValueError(f"unsupported cadence: {var.cadence}")

    if var.unit_scale != 1.0:
        # Preserve attrs across the multiplication so downstream
        # temperature-unit harmonization can still see the original
        # ``units`` attribute. Mark the rescaled output as dimensionless.
        attrs = dict(on_daily.attrs)
        on_daily = on_daily * var.unit_scale
        on_daily.attrs = attrs
        on_daily.attrs["units"] = "1"

    output: dict[str, xr.DataArray] = {}
    if var.kind in ("ocean_surface", "seaice_surface"):
        filled, mask = emit_mask_and_fill(on_daily, fill_iterations=fill_iterations)
        renamed = filled.rename(var.output_name)
        renamed.attrs["original_name"] = var.var_id
        output[var.output_name] = renamed
        output[f"{var.output_name}_mask"] = mask.rename(f"{var.output_name}_mask")
    else:  # atmos_surface (Amon.ts, Eday.ts) — no per-cell mask needed.
        renamed = on_daily.rename(var.output_name)
        renamed.attrs["original_name"] = var.var_id
        output[var.output_name] = renamed
    return output


def emit_mask_and_fill(
    da: xr.DataArray, fill_iterations: int = 50
) -> tuple[xr.DataArray, xr.DataArray]:
    """For an ocean/ice variable with NaN over land (or ice-free
    cells), return ``(filled, mask)`` where ``mask`` is uint8 with the
    valid-cell pattern (per-timestep when ``da`` has a time dim, 2D
    otherwise) and ``filled`` has the NaN regions extrapolated via
    horizontal diffusion so the stored values are NaN-free.

    For time-invariant valid-cell patterns (e.g. ocean-only variables
    where the land mask is static), the mask is collapsed to a single
    2D field to save space; per-timestep masks are used when the
    valid-cell pattern varies in time (e.g. sea-ice extent).
    """
    nan_pattern = da.isnull()
    if "time" in da.dims and nan_pattern.any():
        # Check whether the NaN pattern varies with time.
        first = nan_pattern.isel(time=0)
        time_invariant = bool((nan_pattern == first).all())
    else:
        time_invariant = True

    valid = (~nan_pattern).astype("uint8")
    if time_invariant and "time" in valid.dims:
        valid = valid.isel(time=0).drop_vars("time", errors="ignore")

    # Clear inherited attrs on the mask. ``~nan_pattern`` propagates the
    # source variable's attrs (including ``units = "K"`` once
    # ``harmonize_temperature_to_kelvin`` has run on the parent). If we
    # let the mask carry temperature units, the downstream harmonize
    # loop fires on the mask too and adds 273.15 to the 0/1 values —
    # the on-disk mask ends up holding 273.15 / 274.15 instead of 0/1.
    valid.attrs = {
        "units": "1",
        "long_name": "valid-cell mask (1 = data present, 0 = absent)",
    }

    filled = fill_horizontal_diffuse(da, max_iterations=fill_iterations)
    return filled, valid


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
        # ``end`` is inclusive at day granularity — extend to 23:59:59 so
        # daily timestamps stored at noon (the CMIP6 convention) on the
        # end date are included rather than dropped.
        start_dt = date_type(sy, sm, sd)
        end_dt = date_type(ey, em, ed, 23, 59, 59)
    else:
        start_dt = np.datetime64(start)
        end_dt = np.datetime64(end) + np.timedelta64(1, "D") - np.timedelta64(1, "s")
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
    config. time dim uses ``(chunk_time, shard_time)``; other dims
    are single chunk / single shard (full extent).

    When ``cfg.chunking.variable_batch_size`` is set, data_vars are
    written in batches: first batch ``mode="w"`` creates the store
    (along with the shared coords), subsequent batches ``mode="a"``
    append new data_vars. Each batch is an independent ``to_zarr``
    call, so dask's task graph for one batch is released before the
    next one builds — peak memory stays at
    ``variable_batch_size × per-chunk`` rather than scaling with the
    total number of data_vars in ``ds``. The final consolidated
    metadata pass runs once after all batches are written.
    """
    ds = clear_stale_encoding(ds)

    # Drop scalar non-dim auxiliary coords. CMIP6 source files carry
    # CF-style scalar metadata coords like ``type = "sea_ice"`` or
    # ``height = 2.0`` on individual variables; these survive the
    # regrid+merge path and end up in ``ds.coords``. The batched-write
    # path then breaks on them because xarray re-encodes the same
    # string coord with a different fixed-width dtype between the
    # ``mode="w"`` first batch and the ``mode="a"`` subsequent batches,
    # raising ``ValueError: Mismatched dtypes for variable type``.
    # These coords don't belong in the consolidated output anyway —
    # they were per-source-variable metadata, not whole-dataset
    # facts — so dropping them is the right behaviour.
    scalar_aux_coords = [c for c in ds.coords if c not in ds.dims and ds[c].ndim == 0]
    if scalar_aux_coords:
        ds = ds.drop_vars(scalar_aux_coords)

    chunk_time = cfg.chunking.chunk_time
    shard_time = cfg.chunking.shard_time

    # Align dask chunks on ``time`` with the zarr chunk size. Without
    # this, variables produced by ``causal_monthly_to_daily`` /
    # ``causal_annual_to_daily`` / ``attach_external_forcings`` end up
    # with per-day dask chunks (size 1 along time), and others end up
    # with leap-year boundaries shifting time chunks off ``chunk_time``
    # multiples — zarr v3 refuses to write either pattern ("would
    # overlap multiple Dask chunks ... could lead to corrupted data").
    # ``ds.chunk({"time": chunk_time})`` handles the common cases;
    # ``align_chunks=True`` on ``to_zarr`` is the safety net that
    # rechunks anything still misaligned inside zarr itself. Under
    # the synchronous scheduler, rechunking adjacent chunks holds at
    # most a few extra chunks in memory at once, which for our
    # F22.5-resolution variables is sub-MB.
    if "time" in ds.dims:
        ds = ds.chunk({"time": chunk_time})

    def _encoding_for(var_names: list) -> dict[str, dict]:
        encoding: dict[str, dict] = {}
        for v in var_names:
            var = ds[v]
            chunks = []
            shards: list[int] = []
            for dim, size in zip(var.dims, var.shape):
                if dim == "time":
                    inner = min(chunk_time, size)
                    chunks.append(inner)
                    # Zarr v3 sharding requires the outer (shard) chunk
                    # size to be a multiple of the inner chunk size.
                    # When ``size`` is small enough that
                    # ``min(shard_time, size)`` is less than
                    # ``chunk_time``, or when ``size`` is between two
                    # chunk-sized multiples (e.g. MRI-ESM2-0 ssp245
                    # ships only 5844 timesteps for some variants, and
                    # ``min(7200, 5844) = 5844`` isn't divisible by
                    # ``chunk_time = 360``), we round the requested
                    # shard size down to the nearest multiple of
                    # ``inner`` to keep the codec happy. Trailing
                    # timesteps beyond the last full shard land in a
                    # partial trailing shard, which zarr handles
                    # transparently.
                    requested = min(shard_time or size, size)
                    multiple_of_inner = max(inner, (requested // inner) * inner)
                    shards.append(multiple_of_inner)
                else:
                    chunks.append(size)
                    shards.append(size)
            enc: dict = {"chunks": tuple(chunks)}
            if shard_time is not None and "time" in var.dims:
                enc["shards"] = tuple(shards)
            encoding[v] = enc
        return encoding

    data_vars = list(ds.data_vars)
    batch_size = getattr(cfg.chunking, "variable_batch_size", None)
    n_time = ds.sizes.get("time", 0)
    logging.info(
        "  write_zarr: %d data_vars, batch_size=%s, chunk_time=%d, "
        "shard_time=%s, n_time=%d",
        len(data_vars),
        batch_size,
        chunk_time,
        shard_time,
        n_time,
    )

    if batch_size is None or batch_size >= len(data_vars):
        encoding = _encoding_for(data_vars + list(ds.coords))
        t0 = time.monotonic()
        ds.to_zarr(
            path,
            mode="w",
            encoding=encoding,
            consolidated=True,
            zarr_format=3,
            align_chunks=True,
        )
        logging.info(
            "  write_zarr: single-pass write of %d vars in %.1fs",
            len(data_vars),
            time.monotonic() - t0,
        )
        return

    batches = [
        data_vars[i : i + batch_size] for i in range(0, len(data_vars), batch_size)
    ]
    write_t0 = time.monotonic()
    for i, batch_vars in enumerate(batches):
        sub = ds[batch_vars]
        batch_t0 = time.monotonic()
        rss_before = rss_mib()
        if i == 0:
            sub.to_zarr(
                path,
                mode="w",
                encoding=_encoding_for(batch_vars + list(sub.coords)),
                consolidated=False,
                zarr_format=3,
                align_chunks=True,
            )
        else:
            sub.to_zarr(
                path,
                mode="a",
                encoding=_encoding_for(batch_vars),
                consolidated=False,
                zarr_format=3,
                align_chunks=True,
            )
        rss_after = rss_mib()
        logging.info(
            "  write_zarr: batch %d/%d (%d vars: %s) in %.1fs " "(rss %.0f → %.0f MiB)",
            i + 1,
            len(batches),
            len(batch_vars),
            ",".join(batch_vars),
            time.monotonic() - batch_t0,
            rss_before,
            rss_after,
        )
    # Single consolidated metadata pass after every batch has landed.
    import zarr

    consolidate_t0 = time.monotonic()
    zarr.consolidate_metadata(path)
    logging.info(
        "  write_zarr: total %d batches in %.1fs (consolidate %.2fs)",
        len(batches),
        time.monotonic() - write_t0,
        time.monotonic() - consolidate_t0,
    )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

_EPS = 0.01

_SANITY_RANGES: dict[str, tuple[float, float]] = {
    # Atmospheric daily core — bare CMIP6 names retained for the 3D
    # plev variables (flattened to ``ua1000``..``ua10`` etc.).
    "ua": (-200.0, 200.0),
    "va": (-200.0, 200.0),
    "hus": (-_EPS, 0.05),
    # Mean sea level pressure (Pa) — kept as ``psl`` since the
    # baseline ``PRESsfc`` (surface pressure) is a distinct quantity.
    "psl": (8.0e4, 1.1e5),
    # Non-rename'd optional vars.
    "sfcWind": (-_EPS, 100.0),
    # Surface variables under the baseline naming — see
    # ``CMIP_TO_OUTPUT_RENAMES``.
    "TMP2m": (180.0, 340.0),
    "Q2m": (-_EPS, 0.05),
    "UGRD10m": (-100.0, 100.0),
    "VGRD10m": (-100.0, 100.0),
    "PRATEsfc": (-_EPS, 0.02),  # 1 mm/min cap; tropical convection peaks
    # Radiative fluxes.
    "DSWRFtoa": (-_EPS, 600.0),
    "USWRFtoa": (-_EPS, 600.0),
    "ULWRFtoa": (-_EPS, 400.0),
    "DSWRFsfc": (-_EPS, 600.0),
    "USWRFsfc": (-_EPS, 600.0),
    "DLWRFsfc": (-_EPS, 600.0),
    "ULWRFsfc": (-_EPS, 700.0),
    # Clear-sky radiation (same physical ranges as all-sky).
    "UCSWRFtoa": (-_EPS, 600.0),
    "UCLWRFtoa": (-_EPS, 400.0),
    "DCSWRFsfc": (-_EPS, 600.0),
    "UCSWRFsfc": (-_EPS, 600.0),
    "DCLWRFsfc": (-_EPS, 600.0),
    # Turbulent fluxes.
    "SHTFLsfc": (-1000.0, 1000.0),
    "LHTFLsfc": (-500.0, 1200.0),
    # Land fraction is rescaled to [0, 1] and renamed from ``sftlf``.
    "land_fraction": (-_EPS, 1.0 + _EPS),
    "HGTsfc": (-500.0, 9000.0),  # renamed from ``orog``
    # Geopotential height @ 500 hPa — renamed from ``zg500``.
    "h500": (4500.0, 6100.0),  # Antarctic winter can drop ~4540 m
    # CFday single-pressure-level + 2D diagnostics.
    "TMP700": (220.0, 320.0),
    "PRESsfc": (5.0e4, 1.1e5),
    # ω = Lagrangian air-pressure tendency (Pa/s). Daily means
    # rarely exceed a few Pa/s; allow headroom for storm cells.
    "wap500": (-20.0, 20.0),
    # Cloud water/ice path (kg/m^2). Tropical convection can spike
    # liquid path to a few kg/m^2.
    "clwvi": (-_EPS, 10.0),
    "clivi": (-_EPS, 5.0),
    # Surface-and-ocean variables — source-prefixed output names.
    # Atmospheric surface T (always K post-harmonization).
    "amon_ts": (180.0, 340.0),
    "surface_temperature": (180.0, 340.0),  # renamed from ``eday_ts``
    # Total column water-vapor path (kg/m²). Polar dry-air → ~0,
    # tropical column → ~70; allow some headroom.
    "water_vapor_path": (-_EPS, 100.0),
    # Total water path = vapor + cloud condensate; same shape +
    # roughly the same range (cloud condensate adds <1 kg/m² in the
    # mean, occasional spikes during deep convection).
    "total_water_path": (-_EPS, 110.0),
    # Ocean / sea-ice temperatures (now harmonized to K at ingest).
    "oday_tos": (270.0, 320.0),
    "omon_tob": (250.0, 320.0),
    "simon_sitemptop": (200.0, 280.0),
    "siday_sitemptop": (200.0, 280.0),
    # Sea-ice fractions (rescaled to [0, 1] from CMIP6 % at ingest).
    "simon_sea_ice_fraction": (-_EPS, 1.0 + _EPS),
    "siday_sea_ice_fraction": (-_EPS, 1.0 + _EPS),
    # Derived ocean fraction: 1 - land - sea_ice (clipped).
    "simon_ocean_fraction": (-_EPS, 1.0 + _EPS),
    "siday_ocean_fraction": (-_EPS, 1.0 + _EPS),
    # Sea-ice thickness in metres — Antarctic multi-year ice rarely
    # exceeds 20 m; allow some headroom for ridge models.
    "siday_sithick": (-_EPS, 30.0),
    # Sea-surface salinity (PSU); pure freshwater 0, Dead-Sea-like
    # ocean cells can reach ~40, allow headroom.
    "oday_sos": (-_EPS, 50.0),
    # Sea surface height anomaly (m). Order metres.
    "omon_zos": (-10.0, 10.0),
    # Net downward surface heat flux into ocean (W/m²).
    "omon_hfds": (-1500.0, 1500.0),
    # Mixed layer depth (m). Deep convection regions can exceed 2000 m.
    "omon_mlotst": (-_EPS, 5000.0),
    "oday_omldamax": (-_EPS, 5000.0),
    # NOTE: ``oday_tossq`` is the time-mean SST^2 — units are temperature
    # squared, no simple K conversion (variance), so left out of the
    # range checks.
    # External forcings (input4MIPs / LUH2).
    # CO2 in ppm: pre-industrial ≈ 280, ssp585 endpoint ≈ 2200.
    "input4mips_co2": (250.0, 2500.0),
    # Anthropogenic SO2/BC emission flux (kg m-2 s-1). Industrial-era
    # peaks ~1.5e-9 (SO2) and ~6e-11 (BC); SSP trajectories lower. Allow
    # headroom for cell-level hotspots after sector summation.
    "input4mips_so2": (-_EPS * 1e-12, 5.0e-9),
    "input4mips_bc": (-_EPS * 1e-12, 5.0e-10),
    # LUH2 forest fraction is on [0, 1] by construction.
    "luh2_forest": (-_EPS, 1.0 + _EPS),
}

_DAY_SECONDS = 86400.0
_GLOBAL_MEAN_JUMP_TOL: dict[str, float] = {
    "TMP2m": 2.0,
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


def clamp_static_fractions(ds: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Clip ``sftlf`` to [0, 100] %, rescale to [0, 1], and rename to
    ``land_fraction``.

    Conservative regridding can leave ``sftlf`` slightly above 100% in
    polar rows for some source grids (e.g. CESM2-FV2 up to ~114% in the
    southernmost row). The overshoot is a regridder pole-handling
    artifact, not physical; ``sftlf`` is a fraction with no compensating
    variable in the pipeline, so clipping doesn't break any budget.

    Output is rescaled to [0, 1] and renamed to ``land_fraction`` so the
    convention matches upstream baseline datasets (SHIELD/ERA5 already
    use 0–1 ``land_fraction`` / ``ocean_fraction`` / ``sea_ice_fraction``).
    The original CMIP6 name is preserved in the ``original_name``
    attribute so the provenance back to the source is clear.

    Returns the dataset and a list of warnings naming the worst
    pre-clip overshoot so the regridder defect remains visible.
    """
    warnings: list[str] = []
    if "sftlf" not in ds.data_vars:
        return ds, warnings
    arr = ds["sftlf"]
    vmin = float(arr.min())
    vmax = float(arr.max())
    # The CMIP6 CMOR spec puts ``sftlf`` in percent (0–100), but at
    # least one publisher (FGOALS-f3-L) ships it as a fraction (0–1)
    # in spite of declaring ``units = %``. Detect the input scale via
    # the observed max — anything ≤ 1.0 + _EPS is already a fraction
    # and shouldn't be divided again. Without this check the bad
    # publishers' fields end up at ~0.003 (100× too small), which
    # leaks into the cohort statistics as a clear outlier.
    is_already_fraction = vmax <= 1.0 + _EPS
    if is_already_fraction:
        warnings.append(
            f"sftlf already in [0, 1] (max {vmax:.3g}); skipping /100 rescale"
        )
        land_fraction = arr.clip(0.0, 1.0).rename("land_fraction")
    else:
        if vmax > 100.0 + _EPS or vmin < -_EPS:
            warnings.append(
                f"sftlf clipped to [0, 100]: pre-clip range "
                f"[{vmin:.3g}, {vmax:.3g}]"
            )
        land_fraction = (arr.clip(0.0, 100.0) / 100.0).rename("land_fraction")
    land_fraction.attrs = dict(arr.attrs)
    land_fraction.attrs["original_name"] = "sftlf"
    land_fraction.attrs["units"] = "1"
    ds = ds.drop_vars("sftlf").assign(land_fraction=land_fraction)
    return ds, warnings


def fill_orog_ocean_with_zero(ds: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Fill missing ``orog`` cells with 0 m (sea level).

    Most publishers ship ``orog`` defined everywhere — over the ocean
    the elevation is 0 m. CMCC's family (CMCC-CM2-HR4, CMCC-CM2-SR5,
    CMCC-ESM2) instead publishes ``orog`` only over land cells and
    leaves the ocean as NaN (~63% of the field). The downstream
    pipeline needs orog defined everywhere for the surface-height
    field, so we fill the NaN cells with the correct value (0 m)
    *before* regrid; without this, the conservative regridder produces
    a sparse target field where many cells overlap entirely-NaN
    source cells.

    No-op when ``orog`` is already complete. The caller gets a
    warning entry naming the cell count so the publisher quirk shows
    up in the per-dataset audit trail.
    """
    if "orog" not in ds.data_vars:
        return ds, []
    arr = ds["orog"]
    n_nan = int(arr.isnull().sum())
    if n_nan == 0:
        return ds, []
    filled = arr.fillna(0.0)
    filled.attrs = dict(arr.attrs)
    warning = (
        f"orog: filled {n_nan} NaN cells with 0 m before regrid "
        "(publisher publishes orog only over land cells; ocean = 0 m "
        "elevation at sea level)"
    )
    return ds.assign(orog=filled), [warning]


# Magnitude threshold for ``decode_default_fills``. Any float-valued
# cell whose absolute value exceeds this is treated as a publisher
# fill-value leak and NaN'd. Calibrated to catch every leak observed
# in the v2 cohort without false-positive risk:
#
# - netCDF C default float fill: ``9.969e+36`` (CESM2 omon_tob)
# - 1e+36 raw stored value (BCC-CSM2-MR ua* on the last day, despite
#   ``_FillValue=1e+20`` in metadata — publisher CMOR bug)
# - 1e+35 (FGOALS-f3-L omon_mlotst / omon_tob)
# - 1e+15-1e+16 (MPI-ESM1-2-LR siday_sithick on 4 variants — non-
#   standard fill marker, ``history`` attr says CMOR replaced fills
#   with 1e+20 but a residue at 1e+15 was missed)
#
# 1e10 is at least ~9 orders of magnitude above the largest physical
# value any variable in this dataset takes (CO2 ppm tops out near
# 2500; wind ~200 m/s; everything else smaller), so the threshold has
# effectively zero false-positive surface — there's no physical
# variable that legitimately reaches it.
_FILL_VALUE_THRESHOLD = 1.0e10


def decode_default_fills(
    ds: xr.Dataset, threshold: float = _FILL_VALUE_THRESHOLD
) -> tuple[xr.Dataset, list[str]]:
    """Mask cells whose magnitude exceeds ``threshold`` as NaN, on the
    assumption that they're publisher fill-value leaks.

    Why this exists: xarray's default ``mask_and_scale=True`` replaces
    cells matching ``_FillValue`` / ``missing_value`` with NaN, but
    several CMIP6 publishers ship files where the *declared* fill
    value doesn't match the *stored* bytes (BCC's ``ua`` declares
    1e+20 and stores 1e+36; CESM2's ``omon_tob`` stores 9.97e+36
    without any ``_FillValue`` attribute at all). xarray's decode
    leaves those bad cells in place, and they propagate through
    regridding unchanged because xesmf treats them as ordinary
    numbers. They'd land in training data as ``1e+36`` "wind speeds"
    and blow up the first batch.

    The fix is a magnitude-only check: no physical variable in this
    dataset legitimately exceeds ``threshold``, so anything that does
    is unambiguously a fill marker — regardless of what the publisher
    metadata says.

    Returns ``(ds, warnings)`` where each warning names a variable
    that was masked and the count of cells affected, so the bug is
    visible in the per-dataset audit trail rather than silently
    discarded.
    """
    warnings: list[str] = []
    out_vars: dict[str, xr.DataArray] = {}
    for name, arr in ds.data_vars.items():
        if arr.dtype.kind != "f":
            continue
        # ``abs >= threshold`` rather than ``>``: catches the boundary
        # exactly in case a future publisher uses ``1e10`` as its fill.
        bad_count = int((np.abs(arr) >= threshold).sum())
        if bad_count == 0:
            continue
        cleaned = arr.where(np.abs(arr) < threshold)
        cleaned.attrs = dict(arr.attrs)
        out_vars[name] = cleaned
        vmax = float(np.abs(arr).max())
        warnings.append(
            f"{name}: {bad_count} cells with |value| >= {threshold:g} "
            f"(max |value| {vmax:.3g}); treating as publisher fill-value "
            f"leak and NaN'ing"
        )
    if not out_vars:
        return ds, warnings
    return ds.assign(out_vars), warnings


def run_sanity_checks(ds: xr.Dataset) -> list[str]:
    """Run cheap per-variable range checks plus time-axis continuity.
    Returns a list of human-readable warnings; an empty list means all
    checks passed. Advisory only — out-of-range values are reported
    but not modified (use :func:`decode_default_fills` for the
    fill-value-leak defense).
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
    """Split variables with a ``plev`` dimension into per-level 2D
    variables named ``{var}{hPa}``, e.g. ``ua1000``, ``ua850``, ...,
    ``ua10``. The ``plev`` coordinate is dropped from the returned
    dataset, making all variables uniformly ``(time, lat, lon)`` or
    ``(lat, lon)``.
    """
    plev_vars = [v for v in ds.data_vars if "plev" in ds[v].dims]
    if not plev_vars:
        return ds

    new_vars: dict[str, xr.DataArray] = {}
    plev_pa = ds["plev"].values
    for v in plev_vars:
        da = ds[v]
        for i in range(da.sizes["plev"]):
            label = plev_hpa_label(plev_pa[i])
            level_da = da.isel(plev=i).drop_vars("plev", errors="ignore")
            new_vars[f"{v}{label}"] = level_da

    ds = ds.drop_vars(plev_vars)
    ds = ds.drop_vars(["plev"], errors="ignore")
    return ds.assign(**new_vars)


__all__ = [
    "SimulationBoundaryError",
    "DuplicateTimestampsError",
    "resolve_time_duplicates",
    "validate_cell_methods",
    "grid_fingerprint",
    "normalize_regrid_source",
    "make_regridder",
    "is_unstructured_source",
    "UNSTRUCTURED_METHOD",
    "apply_target_land_mask",
    "apply_output_renames",
    "compute_ocean_fraction",
    "compute_total_water_path",
    "derive_ocean_and_correct_sea_ice",
    "regrid_variables",
    "PLEV8_DEFAULT_HPA",
    "normalize_plev",
    "compute_below_surface_mask",
    "nearest_above_fill",
    "fill_below_surface_smooth",
    "fill_horizontal_diffuse",
    "causal_annual_to_daily",
    "causal_monthly_to_daily",
    "emit_mask_and_fill",
    "finalize_surface_and_ocean_variable",
    "clip_date_for_calendar",
    "apply_time_subset",
    "clear_stale_encoding",
    "write_zarr",
    "clamp_static_fractions",
    "run_sanity_checks",
    "plev_hpa_label",
    "flatten_plev_variables",
    "rss_mib",
    "RssSampler",
]
