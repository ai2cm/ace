"""Shared processing utilities for CMIP6 daily data pipelines.

Data-source-agnostic helpers used by both the Pangeo pipeline
(``process.py``) and the ESGF pipeline (``process_esgf.py``):

- plev normalisation
- below-surface mask computation + nearest-above fill
- derived layer-mean T from the hypsometric equation
- xESMF regridding (with CF-bounds normalisation)
- causal monthly-to-daily mapping for surface-and-ocean variables
- time subsetting + duplicate-timestamp handling
- plev flattening into per-level 2D variables
- sanity checks
- zarr writing with chunk/shard encoding
"""

import logging
import re
from typing import Hashable, Optional

import numpy as np
import xarray as xr
from config import ResolvedDatasetConfig, SurfaceAndOceanVariable

# Physics constants for hypsometric layer-mean T derivation.
R_D = 287.05  # J / (kg K), dry-air gas constant
G = 9.80665  # m / s^2, standard gravity
EPS = 0.608  # (R_v / R_d) - 1, for virtual-to-actual temperature

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
        pieces.append(regridder(ds[vars_], keep_attrs=True))
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
    # Unrecognized units — surface as a warning but leave unchanged.
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
    "PRATEsfc": (-_EPS, 0.01),
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
    "orog": (-500.0, 9000.0),
    # Geopotential height @ 500 hPa — renamed from ``zg500``.
    "h500": (4900.0, 6100.0),
    # Surface-and-ocean variables — source-prefixed output names.
    # Atmospheric surface T (always K post-harmonization).
    "amon_ts": (180.0, 340.0),
    "eday_ts": (180.0, 340.0),
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

_DERIVED_T_RANGE = (150.0, 350.0)

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
    if vmax > 100.0 + _EPS or vmin < -_EPS:
        warnings.append(
            f"sftlf clipped to [0, 100]: pre-clip range " f"[{vmin:.3g}, {vmax:.3g}]"
        )
    land_fraction = (arr.clip(0.0, 100.0) / 100.0).rename("land_fraction")
    land_fraction.attrs = dict(arr.attrs)
    land_fraction.attrs["original_name"] = "sftlf"
    land_fraction.attrs["units"] = "1"
    ds = ds.drop_vars("sftlf").assign(land_fraction=land_fraction)
    return ds, warnings


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

    # Derived layer-mean T comes out of ``flatten_plev_variables`` as
    # ``ta_derived_layer_{lo}_{hi}`` (one per between-plev layer); range
    # check covers any present.
    lo, hi = _DERIVED_T_RANGE
    layer_vars = sorted(v for v in ds.data_vars if v.startswith("ta_derived_layer_"))
    for v in layer_vars:
        arr = ds[v]
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmin < lo or vmax > hi:
            messages.append(
                f"{v} out of range [{lo}, {hi}] K: " f"min={vmin:.2f}, max={vmax:.2f}"
            )

    # Lapse-rate sanity: post-rename 2 m T should be within a few K of
    # the bottom-most derived layer (1000-850 hPa mean).
    bottom_layer = "ta_derived_layer_1000_850"
    if "TMP2m" in ds.data_vars and bottom_layer in ds.data_vars:
        tas_mean = float(ds["TMP2m"].mean())
        layer0_mean = float(ds[bottom_layer].mean())
        diff = tas_mean - layer0_mean
        if abs(diff) > 15.0:
            messages.append(
                f"global mean TMP2m={tas_mean:.2f} K vs {bottom_layer} "
                f"mean={layer0_mean:.2f} K differ by {diff:+.2f} K "
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
    "is_unstructured_source",
    "UNSTRUCTURED_METHOD",
    "apply_target_land_mask",
    "apply_output_renames",
    "compute_ocean_fraction",
    "derive_ocean_and_correct_sea_ice",
    "regrid_variables",
    "PLEV8_DEFAULT_HPA",
    "normalize_plev",
    "compute_below_surface_mask",
    "nearest_above_fill",
    "fill_horizontal_diffuse",
    "compute_derived_layer_T",
    "fill_derived_layer_T",
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
]
