"""
xarray-beam pipeline for the UFS GEFSv13 replay ocean (MOM6) + atmosphere
(FV3) dataset.  Produces a training-ready zarr store for SamudrACE-type
models, matching the runner/infrastructure pattern of ``scripts/era5/``.

The ocean and atmosphere are processed in two independent Beam streams
(one per source dataset, mirroring ``scripts/era5/``) that write to the
same output zarr store.  Because the 3-hourly atmosphere times are
validated up front to exactly interleave the 6-hourly ocean times, the
streams align purely by integer chunk offsets — no per-chunk time
matching is needed.

Ocean stream (per 6-hourly MOM6 time-chunk):

  1. Regrid to a Gaussian grid via xESMF.
  2. Apply thickness-weighted vertical coarsening (``ho``-weighted) and
     split 3-D fields into per-level 2-D variables.
  3. Derive additional variables (sst, ssu/ssv, wfo, hfds, etc.).
  4. Coarsen in time (e.g. 6-hourly → daily).
  5. Insert NaN on land, nearest-neighbour fill residual coastal NaN.

Atmosphere stream (per 3-hourly FV3 time-chunk):

  1. Derive the frozen precipitation rate from bucket accumulations.
  2. Average 3-hourly fields to the 6-hourly ocean cadence.
  3. Regrid to a Gaussian grid via xESMF.
  4. Coarsen in time (e.g. 6-hourly → daily).
  5. Mask sea-ice variables to the ocean.

Data sources::

    gs://noaa-ufs-gefsv13replay/ufs-hr1/{resolution}/{freq}/zarr/

Usage::

    python ufs-replay-pipeline.py <output_path> <start_time> <end_time> \\
        --output_grid F90 --runner DirectRunner

See ``run-dataflow.sh`` and the ``Makefile`` for production invocations.
"""

import argparse
import datetime
import logging
from typing import Sequence

import apache_beam as beam
import cftime as _cftime
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
import xesmf as xe
from apache_beam.options.pipeline_options import PipelineOptions
from obstore.store import from_url
from zarr.storage import ObjectStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCEAN_TIME_STEP = 6  # hours between ocean output timesteps
ATMO_TIME_STEP = 3  # hours between atmosphere output timesteps
DEFAULT_OUTPUT_GRID = "F90"
VDIM = "z_l"  # vertical dimension name after cleaning

# Source zarr URLs (0.25-degree, full ~30-year record 1994–2023)
URL_OCEAN = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/06h-freq/zarr/mom6.zarr"
URL_ATMO = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {
    "F22.5": 22.5,
    "F90": 90,
    "F360": 360,
}

# 3-D ocean variables that get vertically coarsened and split per-level
VARS_3D = ("thetao", "so", "uo", "vo")

# Default vertical coarsening: 75 MOM6 levels → 19 coarse levels matching
# CM4 target depths.  Each [start, end) pair defines a contiguous group.
DEFAULT_VERTICAL_COARSENING_INDICES = [
    [0, 3],
    [3, 8],
    [8, 13],
    [13, 17],
    [17, 20],
    [20, 25],
    [25, 29],
    [29, 33],
    [33, 37],
    [37, 41],
    [41, 44],
    [44, 47],
    [47, 50],
    [50, 53],
    [53, 57],
    [57, 61],
    [61, 66],
    [66, 71],
    [71, 75],
]

# MOM6 variable rename map
OCEAN_RENAME = {"temp": "thetao", "SSH": "zos"}
STRESS_RENAME = {
    "taux": "eastward_surface_wind_stress",
    "tauy": "northward_surface_wind_stress",
}

# MOM6 variables that are dropped entirely
OCEAN_DROP = ["LwLatSens", "pbo"]

# MOM6 surface variables to retain
OCEAN_SURFACE_VARS = ["SSH", "taux", "tauy"]

# FV3 atmosphere forcing variables → output names
ATMO_FORCING_VARS = {
    "dlwrf_ave": "DLWRFsfc",
    "dswrf_ave": "DSWRFsfc",
    "ulwrf_ave": "ULWRFsfc",
    "uswrf_ave": "USWRFsfc",
    "lhtfl_ave": "LHTFLsfc",
    "shtfl_ave": "SHTFLsfc",
    "prateb_ave": "PRATEsfc",
}

# FV3 bucket-accumulated frozen precip variables — converted to a rate
# in the pipeline.  We use the bucket variants (frozrb/tsnowpb) rather
# than the total accumulators (frozr/tsnowp) to keep values small.
# Bucket resets (every fhzero=6h) are detected and handled.
FROZEN_PRECIP_ACCUM_VARS = ["frozrb", "tsnowpb"]

# Sea-ice variables from FV3
ICE_VARS = {"icec": "ocean_sea_ice_fraction", "icetk": "HI"}

# Raw MOM6 flux components used for deriving wfo and hfds, dropped after use
WFO_COMPONENTS = ["evap", "lprec", "fprec", "lrunoff"]
HFDS_COMPONENTS = ["SW", "LW", "latent", "sensible", "Heat_PmE"]


# ---------------------------------------------------------------------------
# Gaussian grid helpers (matching scripts/era5/)
# ---------------------------------------------------------------------------


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes(n: float) -> np.ndarray:
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, _ = leggauss(nlat)
    return np.sort(np.degrees(np.arcsin(x)))


def _make_target_grid(output_grid: str) -> xr.Dataset:
    n = GAUSSIAN_GRID_N[output_grid]
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, 0, 360)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


def _make_source_grid(ds: xr.Dataset) -> xr.Dataset:
    lat = ds["lat"].values
    lon = ds["lon"].values
    if lat[0] > lat[-1]:
        lat = lat[::-1]
    lat_b = _cell_bounds(lat, -90, 90)
    dlon = lon[1] - lon[0] if len(lon) > 1 else 0.25
    lon_b = _cell_bounds(lon, lon[0] - dlon / 2, lon[-1] + dlon / 2)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        }
    )


# ---------------------------------------------------------------------------
# Regridding (cached per worker)
# ---------------------------------------------------------------------------

_REGRIDDER_CACHE = {}


def _get_regridder(output_grid: str, source_grid: xr.Dataset):
    src_shape = (len(source_grid["lat"]), len(source_grid["lon"]))
    cache_key = f"{output_grid}_{src_shape}"
    if cache_key not in _REGRIDDER_CACHE:
        dst = _make_target_grid(output_grid)
        _REGRIDDER_CACHE[cache_key] = xe.Regridder(
            source_grid, dst, "conservative", periodic=True
        )
    return _REGRIDDER_CACHE[cache_key]


def _regrid_dataset(
    ds: xr.Dataset,
    output_grid: str,
    source_grid: xr.Dataset,
    *,
    skipna: bool = True,
    na_thres: float = 1.0,
) -> xr.Dataset:
    """Regrid each variable individually to keep memory bounded."""
    regridder = _get_regridder(output_grid, source_grid)
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    regridded = {}
    for name in ds.data_vars:
        regridded[name] = regridder(
            ds[name], keep_attrs=True, skipna=skipna, na_thres=na_thres
        )
    return xr.Dataset(regridded, attrs=ds.attrs)


# ---------------------------------------------------------------------------
# Vertical coarsening helpers
# ---------------------------------------------------------------------------


def _ocean_weighted_mean(
    da: xr.DataArray, weights: xr.DataArray, dims: str
) -> xr.DataArray:
    """Thickness-weighted mean, masking NaN contributions."""
    weights = weights.drop_attrs()
    valid = da.notnull()
    masked_weights = weights.where(valid, 0.0)
    numerator = (da.fillna(0.0) * masked_weights).sum(dims)
    denominator = masked_weights.sum(dims)
    return numerator / denominator.where(denominator > 0)


def _compute_ocean_vertical_coarsening(
    ds: xr.Dataset,
    vars_3d: Sequence[str],
    interface_indices: Sequence[Sequence[int]],
    vdim: str,
    thickness_name: str = "ho",
) -> xr.Dataset:
    """Thickness-weighted vertical coarsening of 3-D ocean variables.

    Returns a flat Dataset with per-level 2-D fields and passthrough
    2-D variables.  Masks and depth scalars are handled separately
    by ``_extract_invariant_fields`` and ``_build_per_level_masks``.
    """
    coarsened: dict[str, xr.DataArray] = {}

    for i, (start, end) in enumerate(interface_indices):
        ho_slice = ds[thickness_name].isel({vdim: slice(start, end)})

        for name in vars_3d:
            if name not in ds:
                continue
            arr_slice = ds[name].isel({vdim: slice(start, end)})
            coarsened_da = _ocean_weighted_mean(arr_slice, ho_slice, vdim)
            long_name = ds[name].attrs.get("long_name", name)
            coarsened_da.attrs["long_name"] = f"{long_name} level-{i}"
            coarsened[f"{name}_{i}"] = coarsened_da

    # Pass through 2-D variables
    for name in ds.data_vars:
        if name in vars_3d or name == thickness_name:
            continue
        if vdim not in ds[name].dims:
            coarsened[name] = ds[name]

    return xr.Dataset(coarsened)


# ---------------------------------------------------------------------------
# Mask helpers (used by _extract_invariant_fields)
# ---------------------------------------------------------------------------


def _build_3d_mask(
    ds: xr.Dataset, vdim: str, time_idx: int = 0
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build 3-D and 2-D masks from the NaN pattern of a reference variable.

    Returns (mask_3d, mask_2d) where 1 = ocean, 0 = land.
    """
    ref_var = next(
        (v for v in ("thetao", "so") if v in ds and vdim in ds[v].dims),
        None,
    )
    if ref_var is None:
        raise ValueError("Cannot build mask: no 3-D ocean variable found")

    ref_slice = ds[ref_var]
    if "time" in ref_slice.dims:
        ref_slice = ref_slice.isel(time=time_idx)
    mask_3d = (~np.isnan(ref_slice.values)).astype(np.float32)
    mask_3d = xr.DataArray(mask_3d, dims=ref_slice.dims, coords=ref_slice.coords)

    mask_2d = mask_3d.isel({vdim: 0}).astype(np.float32)
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
    return mask_3d, mask_2d


def _build_per_level_masks(
    mask_3d: xr.DataArray,
    vdim: str,
    vertical_coarsening_indices: Sequence[Sequence[int]] | None,
) -> dict[str, xr.DataArray]:
    """Build per-level mask variables from a 3-D mask.

    With coarsening indices, each level mask is the max over the group.
    Without, each level mask is the raw slice.
    """
    masks = {}
    if vertical_coarsening_indices:
        for i, (start, end) in enumerate(vertical_coarsening_indices):
            level_mask = mask_3d.isel({vdim: slice(start, end)}).max(dim=vdim)
            masks[f"mask_{i}"] = level_mask.astype(np.float32)
    else:
        n_levels = mask_3d.sizes[vdim]
        for i in range(n_levels):
            masks[f"mask_{i}"] = mask_3d.isel({vdim: i}).astype(np.float32)
    return masks


# ---------------------------------------------------------------------------
# Nearest-neighbour fill for residual coastal NaN
# ---------------------------------------------------------------------------


def _compute_nn_fill_indices(
    ds: xr.Dataset, mask_2d: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute NN-fill index pairs for variables with coastal NaN.

    Returns a dict mapping variable name → (fill_flat, src_flat) arrays
    of 1-D indices into a (lat*lon) flat view.  Computed once from the
    first timestep; the fill pattern is time-invariant because it derives
    from the static land/ocean mask.
    """
    from scipy.ndimage import distance_transform_edt

    level_prefixes = tuple(f"{v}_" for v in VARS_3D)
    ocean = mask_2d > 0
    fill_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in list(ds.data_vars):
        if name.startswith("mask_") or name.startswith("idepth_"):
            continue
        if name in ("land_fraction", "sea_surface_fraction"):
            continue
        if any(name.startswith(p) for p in level_prefixes):
            continue
        v = ds[name]
        if "time" not in v.dims:
            continue

        sample = v.isel(time=0).values
        need_fill = np.isnan(sample) & ocean
        if not need_fill.any():
            continue

        n_fill = int(need_fill.sum())
        logging.info("NN-fill indices: %d ocean cells for '%s'", n_fill, name)

        valid = ~np.isnan(sample)
        _, nn_idx = distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        src_rows = nn_idx[0][need_fill]
        src_cols = nn_idx[1][need_fill]

        nlat, nlon = sample.shape
        fill_flat = np.ravel_multi_index(np.where(need_fill), (nlat, nlon))
        src_flat = np.ravel_multi_index((src_rows, src_cols), (nlat, nlon))
        fill_map[name] = (fill_flat, src_flat)

    return fill_map


def _apply_nn_fill(
    ds: xr.Dataset,
    fill_map: dict[str, tuple[np.ndarray, np.ndarray]],
) -> xr.Dataset:
    """Apply precomputed NN-fill indices to fill coastal NaN."""
    for name, (fill_flat, src_flat) in fill_map.items():
        if name not in ds:
            continue
        v = ds[name]
        data = v.values
        flat = data.reshape(data.shape[0], -1)
        flat[:, fill_flat] = flat[:, src_flat]
        ds[name] = xr.DataArray(data, dims=v.dims, coords=v.coords, attrs=v.attrs)
    return ds


# ---------------------------------------------------------------------------
# Data opening
# ---------------------------------------------------------------------------


def _make_zarr_store(url: str, read_only: bool = True):
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    else:
        return url


def _match_time_type(dt, reference_time):
    """Convert *dt* to match the type of *reference_time*.

    CLI arguments are parsed as ``datetime.datetime`` but the zarr stores
    use cftime objects.  This converts so that ``xr.sel(time=slice(...))``
    works correctly.
    """
    if isinstance(reference_time, _cftime.datetime):
        # Match the exact cftime subclass used in the dataset
        cf_cls = type(reference_time)
        if isinstance(dt, cf_cls):
            return dt
        if isinstance(dt, (pd.Timestamp, datetime.datetime)):
            return cf_cls(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        if isinstance(dt, _cftime.datetime):
            return cf_cls(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    # numpy datetime64 / pd.Timestamp index — just use pd.Timestamp
    if isinstance(dt, (pd.Timestamp, datetime.datetime)):
        return pd.Timestamp(dt)
    return dt


def _open_ufs_zarr(url: str) -> xr.Dataset:
    """Open a UFS zarr store with cftime support and suppress warnings."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ds = xr.open_zarr(
            _make_zarr_store(url),
            chunks=None,
            use_cftime=True,
        )
    return ds


def open_ocean(variables: list[str], start_time, end_time) -> xr.Dataset:
    """Open ocean variables from MOM6 zarr store (lazy, no Dask)."""
    ds = _open_ufs_zarr(URL_OCEAN)
    ds = ds[[v for v in variables if v in ds]]
    ref = ds.time.values[0]
    logging.info("Ocean time type: %s (%s)", type(ref).__name__, ref)
    t0 = _match_time_type(start_time, ref)
    t1 = _match_time_type(end_time, ref)
    ds = ds.sel(time=slice(t0, t1))
    return ds


def open_atmo(variables: list[str], start_time, end_time) -> xr.Dataset:
    """Open atmosphere variables from FV3 zarr store (lazy, no Dask)."""
    ds = _open_ufs_zarr(URL_ATMO)
    # Rename horizontal dims to match ocean convention
    rename = {}
    if "grid_xt" in ds.dims:
        rename["grid_xt"] = "lon"
    if "grid_yt" in ds.dims:
        rename["grid_yt"] = "lat"
    if rename:
        ds = ds.rename(rename)
    found = [v for v in variables if v in ds]
    missing = [v for v in variables if v not in ds]
    if missing:
        logging.warning("Atmo vars not found in store: %s", missing)
    logging.info("Atmo vars found: %s", found)

    # The FV3 store may use a time dimension called "time" on some
    # variables but have other variables on different time dims (e.g.
    # "ftime").  Select variables first, then log the time coordinate.
    ds = ds[found]
    logging.info("Atmo dims after var selection: %s", dict(ds.sizes))

    ref = ds.time.values[0]
    logging.info("Atmo time type: %s (%s)", type(ref).__name__, ref)
    t0 = _match_time_type(start_time, ref)
    t1 = _match_time_type(end_time, ref)
    logging.info("Atmo time slice: %s to %s", t0, t1)
    ds = ds.sel(time=slice(t0, t1))
    logging.info("Atmo after time slice: %d timesteps", ds.sizes["time"])

    # Drop stray non-dimension coordinates (cftime, ftime, etc.) that
    # pollute downstream coarsen steps and the output encoding.
    keep_coords = {"time", "lat", "lon"}
    stray = [c for c in ds.coords if c not in keep_coords and c not in ds.dims]
    if stray:
        ds = ds.drop_vars(stray)
    return ds


def _clean_ocean_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Clean UFS MOM6 ocean dataset right after opening.

    Normalizes vertical dimensions, drops unused variables, renames
    to standard names, and removes stray non-dimension coordinates.
    This runs once at the front of the pipeline so that every downstream
    function receives a consistently shaped dataset.
    """
    # MOM6's layer thickness 'ho' lives on the 'zl' dimension while all
    # tracer variables (thetao, so, …) use 'z_l'.  Move ho onto z_l so
    # that the rest of the pipeline only has to deal with one vertical dim.
    if "zl" in ds.dims and "ho" in ds and "zl" in ds["ho"].dims:
        ho = ds["ho"].rename({"zl": VDIM})
        ho = ho.assign_coords({VDIM: ds[VDIM].values})
        ds = ds.drop_vars("ho")
        ds["ho"] = ho
        ds = ds.drop_dims("zl", errors="ignore")

    ds = ds.drop_vars(
        [v for v in OCEAN_DROP if v in ds] + ["landsea_mask"],
        errors="ignore",
    )

    rename_map = {k: v for k, v in OCEAN_RENAME.items() if k in ds}
    rename_map.update({k: v for k, v in STRESS_RENAME.items() if k in ds})
    if rename_map:
        ds = ds.rename(rename_map)

    # Drop stray non-dimension coordinates (cftime, ftime, etc.) that
    # pollute downstream merge/coarsen steps.
    keep_coords = {"time", "lat", "lon", VDIM}
    stray = [c for c in ds.coords if c not in keep_coords and c not in ds.dims]
    if stray:
        ds = ds.drop_vars(stray)

    return ds


def _validate_time_alignment(ocean_times, atmo_times) -> None:
    """Check that the atmosphere times exactly interleave the ocean times.

    The 6-hourly ocean fields are averages centered on their labels,
    while the 3-hourly FV3 interval-average fields are labeled at the
    END of their averaging window, so ocean time T is covered by the
    atmosphere pair (T, T+3h).  Verifying this correspondence once up
    front lets the two Beam streams align purely by integer chunk
    offsets, with no per-chunk time matching.
    """
    if len(atmo_times) != 2 * len(ocean_times):
        raise ValueError(
            f"Expected exactly {2 * len(ocean_times)} atmosphere timesteps "
            f"(2 per ocean timestep), got {len(atmo_times)}."
        )
    step = datetime.timedelta(hours=ATMO_TIME_STEP)
    for i, ocean_time in enumerate(ocean_times):
        if (
            atmo_times[2 * i] != ocean_time
            or atmo_times[2 * i + 1] != ocean_time + step
        ):
            raise ValueError(
                f"Atmosphere times ({atmo_times[2 * i]}, {atmo_times[2 * i + 1]}) "
                f"do not cover ocean time {ocean_time}; expected "
                f"({ocean_time}, {ocean_time + step})."
            )


# ---------------------------------------------------------------------------
# Per-chunk processing (called by each Beam worker)
# ---------------------------------------------------------------------------

# Source grids are built lazily and cached per worker
_SOURCE_GRID_CACHE = {}


def _get_source_grid(ds: xr.Dataset) -> xr.Dataset:
    """Build and cache a source grid descriptor."""
    key = (len(ds["lat"]), len(ds["lon"]))
    if key not in _SOURCE_GRID_CACHE:
        _SOURCE_GRID_CACHE[key] = _make_source_grid(ds)
    return _SOURCE_GRID_CACHE[key]


def _finalize_chunk(ds: xr.Dataset) -> xr.Dataset:
    """Final cleanup shared by the ocean and atmosphere streams.

    Drops non-output dimensions and stray coordinates, casts to float32,
    and drops scalar and time-invariant variables — xarray-beam's
    ConsolidateChunks requires all variables to share the same
    dimensions, and the invariant fields are written once during
    template creation rather than streamed through the pipeline.
    """
    keep_dims = {"time", "lat", "lon"}
    drop_dims = [d for d in ds.dims if d not in keep_dims]
    if drop_dims:
        ds = ds.drop_dims(drop_dims)
    ds = ds.reset_coords(drop=True)

    for name in ds.data_vars:
        if ds[name].dtype not in (np.float32, np.int32):
            ds[name] = ds[name].astype(np.float32)

    time_varying = [
        name for name in ds.data_vars if set(ds[name].dims) == {"time", "lat", "lon"}
    ]
    return ds[time_varying]


# ---------------------------------------------------------------------------
# Ocean stream (6-hourly MOM6 data)
# ---------------------------------------------------------------------------


def _process_ocean_chunk(
    ds_ocean: xr.Dataset,
    output_grid: str,
    vertical_coarsening_indices: Sequence[Sequence[int]],
    time_coarsen_factor: int,
    source_grid_ocean: xr.Dataset,
    invariant_ds: xr.Dataset,
    nn_fill_map: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> xr.Dataset:
    """Process one time-chunk of ocean data.

    Receives a loaded (in-memory) Dataset for a small number of
    timesteps, plus the precomputed invariant fields (masks, fractions)
    and NN-fill indices, and returns the processed output Dataset.
    """
    xr.set_options(keep_attrs=True)

    # "depth" is only needed for the invariant deptho field, which is
    # produced during template creation.
    ds_ocean = ds_ocean.drop_vars("depth", errors="ignore")

    # Separate ho for independent regridding (thickness-weighted
    # coarsening needs ho regridded without NaN-masking influence
    # from tracer fields that have deeper NaN patterns).
    if "ho" not in ds_ocean:
        raise ValueError("'ho' is required for thickness-weighted coarsening")
    ho_ds = ds_ocean[["ho"]]
    ds_ocean = ds_ocean.drop_vars("ho")

    # --- Horizontal regridding ---
    ds_ocean = _regrid_dataset(
        ds_ocean,
        output_grid,
        source_grid_ocean,
        skipna=True,
        na_thres=1.0,
    )
    ho_ds = _regrid_dataset(
        ho_ds,
        output_grid,
        source_grid_ocean,
        skipna=True,
        na_thres=1.0,
    )
    ds_ocean["ho"] = ho_ds["ho"].assign_coords(
        lat=ds_ocean.lat.values, lon=ds_ocean.lon.values
    )

    # --- Vertical coarsening ---
    vars_3d_present = [v for v in VARS_3D if v in ds_ocean]
    indices_as_tuples = [tuple(pair) for pair in vertical_coarsening_indices]
    ds = _compute_ocean_vertical_coarsening(
        ds_ocean,
        vars_3d_present,
        indices_as_tuples,
        VDIM,
    )

    # --- Derived variables ---
    # SST in Kelvin
    if "thetao_0" in ds:
        sst_K = ds["thetao_0"] + 273.15
        sst_K.attrs = {"long_name": "Sea surface temperature", "units": "K"}
        ds["sst"] = sst_K
    if "zos" in ds:
        ds["zos"].attrs.setdefault("long_name", "Sea Surface Height")

    # Surface velocity aliases
    if "uo_0" in ds:
        ssu = ds["uo_0"]
        ssu.attrs = {"long_name": "Sea surface x-velocity", "units": "m/s"}
        ds["ssu"] = ssu
    if "vo_0" in ds:
        ssv = ds["vo_0"]
        ssv.attrs = {"long_name": "Sea surface y-velocity", "units": "m/s"}
        ds["ssv"] = ssv

    # Stress aliases
    if "eastward_surface_wind_stress" in ds:
        tauuo = ds["eastward_surface_wind_stress"]
        tauuo.attrs = {"long_name": "Surface Downward X Stress", "units": "N/m2"}
        ds["tauuo"] = tauuo
    if "northward_surface_wind_stress" in ds:
        tauvo = ds["northward_surface_wind_stress"]
        tauvo.attrs = {"long_name": "Surface Downward Y Stress", "units": "N/m2"}
        ds["tauvo"] = tauvo

    # wfo: water flux = evap + lprec + fprec + lrunoff
    if all(v in ds for v in WFO_COMPONENTS):
        wfo = sum(ds[c] for c in WFO_COMPONENTS)
        wfo.attrs = {
            "long_name": "Water Flux Into Sea Water",
            "units": "kg/(m2 s)",
        }
        ds["wfo"] = wfo

    # hfds: net surface heat flux
    if all(v in ds for v in HFDS_COMPONENTS):
        hfds = sum(ds[c] for c in HFDS_COMPONENTS)
        hfds.attrs = {
            "long_name": "Downward Heat Flux at Sea Water Surface",
            "units": "W/m2",
        }
        ds["hfds"] = hfds

    # Drop raw flux components
    ds = ds.drop_vars(
        [v for v in WFO_COMPONENTS + HFDS_COMPONENTS if v in ds],
        errors="ignore",
    )

    # --- Time coarsening ---
    # coarsen().mean() also averages the time coordinate.  The ocean 6h
    # fields are center-labeled (00Z, 06Z, 12Z, 18Z) covering
    # 21Z(-1d)–21Z, so the daily-mean label lands naturally at 09Z.
    # The atmosphere stream produces identical labels (see
    # _process_atmo_chunk), so both streams share one output time
    # coordinate.
    if time_coarsen_factor > 1:
        ds = ds.coarsen(time=time_coarsen_factor, boundary="trim").mean()

    # --- Insert NaN on land ---
    # Vertically coarsened per-level fields use the matching per-level
    # mask; everything else uses the surface mask.
    for name in list(ds.data_vars):
        level = name.rsplit("_", 1)[-1]
        mask_name = f"mask_{level}" if level.isdigit() else "mask_2d"
        ds[name] = ds[name].where(invariant_ds[mask_name] > 0)

    # Derived post-masking variables
    if "hfds" in ds:
        ds["hfds_total_area"] = ds["hfds"] * invariant_ds["sea_surface_fraction"]
        ds["hfds_total_area"].attrs = {
            "long_name": "heat flux into sea water scaled by sea surface fraction",
            "units": "W/m2",
        }

    # --- NN fill for residual coastal NaN ---
    if nn_fill_map:
        ds = _apply_nn_fill(ds, nn_fill_map)

    return _finalize_chunk(ds)


def process_ocean_chunk(
    key,
    ds_ocean_chunk,
    output_grid=DEFAULT_OUTPUT_GRID,
    vertical_coarsening_indices=None,
    time_coarsen_factor=1,
    invariant_ds=None,
    nn_fill_map=None,
):
    """Beam-compatible ocean stream function: (key, ds) → (new_key, output_ds).

    Called by ``beam.MapTuple`` for each time-chunk of ocean data.  Each
    output timestep consumes ``time_coarsen_factor`` 6-hourly ocean
    timesteps, so the output time offset is the input offset divided by
    that factor.
    """
    if vertical_coarsening_indices is None:
        vertical_coarsening_indices = DEFAULT_VERTICAL_COARSENING_INDICES

    logging.info("Processing ocean chunk at key=%s", key)
    ds_ocean_chunk = ds_ocean_chunk.load()

    output = _process_ocean_chunk(
        ds_ocean_chunk,
        output_grid,
        vertical_coarsening_indices,
        time_coarsen_factor,
        _get_source_grid(ds_ocean_chunk),
        invariant_ds,
        nn_fill_map,
    )

    new_key = key.replace(
        offsets={"time": key.offsets["time"] // time_coarsen_factor},
        vars=frozenset(output.keys()),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Atmosphere stream (3-hourly FV3 data)
# ---------------------------------------------------------------------------


def _process_atmo_chunk(
    ds_atmo: xr.Dataset,
    output_grid: str,
    time_coarsen_factor: int,
    source_grid_atmo: xr.Dataset,
    invariant_ds: xr.Dataset,
    nn_fill_map: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> xr.Dataset:
    """Process one time-chunk of atmosphere data.

    Receives a loaded (in-memory) Dataset whose 3-hourly times pairwise
    cover the 6-hourly ocean times of the corresponding output chunk
    (validated up front by ``_validate_time_alignment``).
    """
    xr.set_options(keep_attrs=True)

    # Derive frozen precipitation rate from bucket-accumulated fields
    # (frozrb = graupel, tsnowpb = snow).  The bucket empties every
    # output step (every 3h), so each value IS the 3h accumulation.
    # Simply divide by dt to get the rate — no differencing needed.
    accum_vars = [v for v in FROZEN_PRECIP_ACCUM_VARS if v in ds_atmo]
    if accum_vars:
        dt_seconds = ATMO_TIME_STEP * 3600.0  # 3h → seconds
        frozen_accum = sum(ds_atmo[v] for v in accum_vars)
        frozen_rate = (frozen_accum / dt_seconds).clip(min=0)
        frozen_rate.attrs = {
            "long_name": "total frozen precipitation rate",
            "units": "kg/m**2/s",
        }
        ds_atmo = ds_atmo.drop_vars(accum_vars)
        ds_atmo["total_frozen_precipitation_rate"] = frozen_rate

    # Average consecutive pairs of 3h fields → 6h, labeled with the
    # first timestamp of each pair.  The FV3 interval-average fields are
    # labeled at the END of their 3h averaging window, so the pair
    # (T, T+3h) covers T-3h through T+3h — exactly the window of the
    # center-labeled 6h ocean average at T.  coarsen is preferred over
    # resample because the atmo data is regularly spaced and coarsen
    # always produces equal-sized groups (resample can create uneven
    # bins at boundaries).
    atmo_times = ds_atmo.time.values
    ds = ds_atmo.coarsen(time=2, boundary="trim").mean()
    ds = ds.assign_coords(time=atmo_times[::2])

    # --- Horizontal regridding ---
    ds = _regrid_dataset(
        ds,
        output_grid,
        source_grid_atmo,
        skipna=True,
        na_thres=1.0,
    )
    ds = ds.assign_coords(lat=invariant_ds.lat.values, lon=invariant_ds.lon.values)

    # --- Time coarsening ---
    # The 6h labels above equal the ocean times, so coarsening averages
    # the same label groups as the ocean stream and both streams emit
    # identical output time coordinates.
    if time_coarsen_factor > 1:
        ds = ds.coarsen(time=time_coarsen_factor, boundary="trim").mean()

    # Rename to output names
    rename_map = {**ATMO_FORCING_VARS, **ICE_VARS}
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})

    # Sea-ice variables are only meaningful over the ocean.  The
    # atmospheric forcing variables are valid globally (FV3 is a global
    # model) and must NOT be masked — downstream training configs rely
    # on having values everywhere including over land.
    mask_2d = invariant_ds["mask_2d"]
    if "ocean_sea_ice_fraction" in ds:
        ds["ocean_sea_ice_fraction"] = ds["ocean_sea_ice_fraction"].where(mask_2d > 0)
    if "HI" in ds:
        ds["HI"] = ds["HI"].where(mask_2d > 0)
        if "ocean_sea_ice_fraction" in ds:
            ds["HI"] = ds["HI"].where(ds["ocean_sea_ice_fraction"] > 0, 0.0)
        # Re-apply NaN on land — the ice-free zeroing above converts land
        # from NaN to 0.0 (because NaN > 0 is False), but ACE's output
        # masker expects NaN on land to match the target NaN pattern.
        ds["HI"] = ds["HI"].where(mask_2d > 0)
        siv = ds["HI"]
        siv.attrs = {"long_name": "Sea Ice Volume Per Area", "units": "m"}
        ds["sea_ice_volume"] = siv

    # --- NN fill for residual coastal NaN ---
    if nn_fill_map:
        ds = _apply_nn_fill(ds, nn_fill_map)

    return _finalize_chunk(ds)


def process_atmo_chunk(
    key,
    ds_atmo_chunk,
    output_grid=DEFAULT_OUTPUT_GRID,
    time_coarsen_factor=1,
    invariant_ds=None,
    nn_fill_map=None,
):
    """Beam-compatible atmosphere stream function: (key, ds) → (new_key, output_ds).

    Called by ``beam.MapTuple`` for each time-chunk of atmosphere data.
    Each output timestep consumes ``2 * time_coarsen_factor`` 3-hourly
    atmosphere timesteps (pair-averaging to 6-hourly, then coarsening),
    so the output time offset is the input offset divided by that.
    """
    logging.info("Processing atmo chunk at key=%s", key)
    ds_atmo_chunk = ds_atmo_chunk.load()

    output = _process_atmo_chunk(
        ds_atmo_chunk,
        output_grid,
        time_coarsen_factor,
        _get_source_grid(ds_atmo_chunk),
        invariant_ds,
        nn_fill_map,
    )

    new_key = key.replace(
        offsets={"time": key.offsets["time"] // (2 * time_coarsen_factor)},
        vars=frozenset(output.keys()),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Template building
# ---------------------------------------------------------------------------


def _extract_invariant_fields(
    ds_ocean: xr.Dataset,
    output_grid: str,
    vertical_coarsening_indices: Sequence[Sequence[int]],
    source_grid_ocean: xr.Dataset,
) -> xr.Dataset:
    """Extract time-invariant fields from the ocean data.

    Returns a Dataset containing scalar fields (idepth_*) and 2-D
    spatial fields (mask_*, land_fraction, sea_surface_fraction, deptho)
    without a time dimension.  These are written once to the zarr store
    during template creation, not streamed through the Beam pipeline,
    and are also passed to every Beam worker for masking.
    """
    invariant = {}

    # idepth scalars: level interfaces (N+1 boundaries for N layers).
    # ACE's DepthCoordinate expects idepth[i] to be the upper boundary
    # of layer i, with idepth[N] being the bottom of the last layer.
    depths = ds_ocean[VDIM].values
    invariant["idepth_0"] = xr.DataArray(
        0.0,
        attrs={"units": "meters", "long_name": "Depth interface 0 (surface)"},
    )
    for i, (start, end) in enumerate(vertical_coarsening_indices):
        bottom_depth = float(depths[end - 1])
        invariant[f"idepth_{i + 1}"] = xr.DataArray(
            bottom_depth,
            attrs={"units": "meters", "long_name": f"Depth interface {i + 1}"},
        )

    # 2-D spatial invariant fields: derive from a single ocean timestep.
    ds_ocean_1t = ds_ocean.isel(time=0).load()

    # Build the native-resolution binary mask BEFORE regridding.
    # Conservatively regridding this binary mask gives fractional
    # sea_surface_fraction at coastal cells (e.g., 0.7 if 70% of the
    # 0.25° source cells within a 1° target cell are ocean).
    native_mask_3d, native_mask_2d = _build_3d_mask(ds_ocean_1t, VDIM)

    if output_grid:
        # Regrid native binary mask → fractional ocean coverage
        native_mask_ds = xr.Dataset({"mask_2d": native_mask_2d})
        frac_ds = _regrid_dataset(
            native_mask_ds,
            output_grid,
            source_grid_ocean,
            skipna=False,
            na_thres=1.0,
        )
        sea_fraction = frac_ds["mask_2d"].clip(0, 1).astype(np.float32)

        # Regrid ocean data for building binary masks from NaN pattern
        ds_ocean_1t = _regrid_dataset(
            ds_ocean_1t,
            output_grid,
            source_grid_ocean,
            skipna=True,
            na_thres=1.0,
        )
    else:
        sea_fraction = native_mask_2d

    # Binary masks for NaN insertion (from regridded NaN pattern)
    mask_3d, mask_2d = _build_3d_mask(ds_ocean_1t, VDIM)
    level_masks = _build_per_level_masks(mask_3d, VDIM, vertical_coarsening_indices)
    invariant.update(level_masks)
    invariant["mask_2d"] = mask_2d

    # Fractional land/sea fractions from regridded native mask
    land_frac = (1.0 - sea_fraction).astype(np.float32)
    land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
    sea_fraction.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
    invariant["land_fraction"] = land_frac
    invariant["sea_surface_fraction"] = sea_fraction

    # deptho
    depth_name = "depth" if "depth" in ds_ocean_1t else "deptho"
    if depth_name in ds_ocean_1t:
        deptho = ds_ocean_1t[depth_name].astype(np.float32)
        deptho.attrs = {"long_name": "Sea Floor Depth Below Geoid", "units": "m"}
        invariant["deptho"] = deptho

    # Drop stray scalar coordinates (e.g. z_l left over from the isel in
    # _build_3d_mask) so they don't leak into the output via the template.
    return xr.Dataset(invariant).reset_coords(drop=True)


def _make_template(
    ds_ocean: xr.Dataset,
    ds_atmo: xr.Dataset,
    output_grid: str,
    vertical_coarsening_indices: Sequence[Sequence[int]],
    time_coarsen_factor: int,
    output_time: list,
) -> tuple[
    xr.Dataset,
    xr.Dataset,
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Eagerly process one output timestep to build the output zarr template.

    Returns (template, invariant_ds, ocean_nn_fill_map, atmo_nn_fill_map)
    where the invariant fields and per-stream NN-fill indices are
    precomputed data to be reused by every Beam worker.
    """
    logging.info("Building template from first output timestep")

    # Extract scalar (idepth_*) and 2-D spatial (mask_*, land_fraction,
    # etc.) invariant fields — all without a time dimension.
    src_ocean = _make_source_grid(ds_ocean.isel(time=0).load())
    invariant_ds = _extract_invariant_fields(
        ds_ocean,
        output_grid,
        vertical_coarsening_indices,
        src_ocean,
    ).drop_encoding()
    mask_2d_arr = invariant_ds["mask_2d"].values

    # Process one chunk of each stream to get the time-varying variable
    # schema.  Process WITHOUT NN fill so the fill indices can be
    # extracted from the un-filled data (the fill pattern depends only
    # on the static ocean mask, so it is computed once here and passed
    # to every Beam worker), then apply the fill.
    ocean_per_output = max(1, time_coarsen_factor)
    ds_ocean_small = ds_ocean.isel(time=slice(0, ocean_per_output)).load()
    processed_ocean = _process_ocean_chunk(
        ds_ocean_small,
        output_grid,
        vertical_coarsening_indices,
        time_coarsen_factor,
        src_ocean,
        invariant_ds,
        nn_fill_map=None,
    ).drop_encoding()
    ocean_nn_fill_map = _compute_nn_fill_indices(processed_ocean, mask_2d_arr)
    processed_ocean = _apply_nn_fill(processed_ocean, ocean_nn_fill_map)

    # Atmosphere sample: two 3-hourly timesteps per 6-hourly ocean timestep.
    ds_atmo_small = ds_atmo.isel(time=slice(0, 2 * ocean_per_output)).load()
    processed_atmo = _process_atmo_chunk(
        ds_atmo_small,
        output_grid,
        time_coarsen_factor,
        _make_source_grid(ds_atmo_small),
        invariant_ds,
        nn_fill_map=None,
    ).drop_encoding()
    atmo_nn_fill_map = _compute_nn_fill_indices(processed_atmo, mask_2d_arr)
    processed_atmo = _apply_nn_fill(processed_atmo, atmo_nn_fill_map)

    processed = xr.merge([processed_ocean, processed_atmo])

    # Squeeze out the single-timestep time dim, then re-expand with the
    # full output time coordinate (same pattern as ERA5 pipeline).
    processed = processed.squeeze("time", drop=True)
    template = xbeam.make_template(processed)
    template = template.expand_dims(dim={"time": output_time}, axis=0)

    # Add invariant fields to template (written once, no time dimension)
    for name in invariant_ds.data_vars:
        if name not in template:
            template[name] = invariant_ds[name]

    return template, invariant_ds, ocean_nn_fill_map, atmo_nn_fill_map


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _get_parser():
    parser = argparse.ArgumentParser(
        description="UFS GEFSv13 replay ocean preprocessing pipeline"
    )
    parser.add_argument(
        "output_path", type=str, help="Output path for the processed zarr dataset"
    )
    parser.add_argument("start_time", type=str, help="Start of output dataset (ISO)")
    parser.add_argument("end_time", type=str, help="End of output dataset (ISO)")
    parser.add_argument(
        "--output_grid",
        type=str,
        default="F90",
        help="Output grid: 'F90' for 1°, 'F22.5' for 4°.",
    )
    parser.add_argument(
        "--output_time_chunksize",
        type=int,
        default=1,
        help="Number of times per output chunk (zarr inner chunk).",
    )
    parser.add_argument(
        "--output_time_shardsize",
        type=int,
        default=360,
        help="Number of times per output shard (zarr shard size).",
    )
    parser.add_argument(
        "--process_time_chunksize",
        type=int,
        default=4,
        help=(
            "Number of ocean timesteps per processing chunk.  Should be a "
            "multiple of time_coarsen_factor.  With time_coarsen_factor=4 "
            "the default of 4 produces 1 output timestep per chunk."
        ),
    )
    parser.add_argument(
        "--time_coarsen_factor",
        type=int,
        default=4,
        help="Temporal coarsening factor (e.g. 4 for 6h→daily). Default: 4.",
    )
    parser.add_argument(
        "--vertical_coarsening_indices",
        type=str,
        default=None,
        help=(
            "JSON-encoded list of [start,end) pairs for vertical coarsening. "
            "Default uses the built-in 75→19 level mapping."
        ),
    )
    return parser


def main():
    import json

    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()
    logging.info("Pipeline args: %s", pipeline_args)

    start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
    end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%dT%H:%M:%S")

    assert (
        start_time.hour % OCEAN_TIME_STEP == 0
    ), f"start_time hour must be a multiple of {OCEAN_TIME_STEP}"
    assert (
        end_time.hour % OCEAN_TIME_STEP == 0
    ), f"end_time hour must be a multiple of {OCEAN_TIME_STEP}"

    # Parse vertical coarsening indices
    if args.vertical_coarsening_indices is not None:
        vert_indices = json.loads(args.vertical_coarsening_indices)
    else:
        vert_indices = DEFAULT_VERTICAL_COARSENING_INDICES

    time_coarsen_factor = args.time_coarsen_factor

    # Validate chunk/shard divisibility
    msg = (
        "output_time_shardsize must be a multiple of output_time_chunksize, "
        f"got {args.output_time_shardsize} and {args.output_time_chunksize}"
    )
    assert args.output_time_shardsize % args.output_time_chunksize == 0, msg

    if time_coarsen_factor > 1:
        assert args.process_time_chunksize % time_coarsen_factor == 0, (
            f"process_time_chunksize ({args.process_time_chunksize}) must be "
            f"a multiple of time_coarsen_factor ({time_coarsen_factor})"
        )

    output_chunks = {"time": args.output_time_chunksize}
    output_shards = {"time": args.output_time_shardsize}
    process_chunks = {"time": args.process_time_chunksize}

    # --- Open source datasets ---
    logging.info("Opening datasets")

    # Determine which ocean variables to load (use source names, not
    # post-rename names — the MOM6 store has "temp" not "thetao")
    ocean_source_3d = list(OCEAN_RENAME.keys()) + [
        v for v in VARS_3D if v not in OCEAN_RENAME.values()
    ]
    ocean_surface = list(OCEAN_SURFACE_VARS)
    ocean_stress = list(STRESS_RENAME.keys())
    ocean_flux_vars = WFO_COMPONENTS + HFDS_COMPONENTS
    ocean_load_vars = (
        ocean_source_3d
        + ["ho", "depth"]
        + ocean_surface
        + ocean_stress
        + ocean_flux_vars
    )
    # Deduplicate
    ocean_load_vars = list(dict.fromkeys(ocean_load_vars))

    ds_ocean = open_ocean(ocean_load_vars, start_time, end_time)
    # Truncate to a multiple of process_time_chunksize so every chunk has
    # exactly the same number of timesteps (avoids coarsen failures).
    n_ocean = ds_ocean.sizes["time"]
    n_usable = (n_ocean // args.process_time_chunksize) * args.process_time_chunksize
    if n_usable < n_ocean:
        logging.warning(
            "Trimming ocean time from %d to %d timesteps to be divisible "
            "by process_time_chunksize=%d (dropping last %d timesteps)",
            n_ocean,
            n_usable,
            args.process_time_chunksize,
            n_ocean - n_usable,
        )
        ds_ocean = ds_ocean.isel(time=slice(0, n_usable))
    ds_ocean = _clean_ocean_dataset(ds_ocean)
    logging.info("Ocean dataset: %s", dict(ds_ocean.sizes))

    # Atmosphere: 3-hourly data covering the same interval as the ocean.
    # Ocean time T (a center-labeled 6h average) is covered by the
    # end-labeled 3h atmosphere fields at T and T+3h, so the atmosphere
    # range extends one atmosphere timestep past the last ocean time.
    ocean_times = ds_ocean.time.values
    atmo_load_vars = (
        list(ATMO_FORCING_VARS.keys())
        + list(ICE_VARS.keys())
        + FROZEN_PRECIP_ACCUM_VARS
    )
    atmo_end = ocean_times[-1] + datetime.timedelta(hours=ATMO_TIME_STEP)
    ds_atmo = open_atmo(atmo_load_vars, ocean_times[0], atmo_end)
    logging.info("Atmo dataset: %s", dict(ds_atmo.sizes))

    # With aligned times, the two Beam streams correspond purely by
    # integer chunk offsets: ocean offset i and atmosphere offset 2*i
    # contribute to output offset i // time_coarsen_factor.
    _validate_time_alignment(ocean_times, ds_atmo.time.values)

    # --- Output time coordinate ---
    # Let xarray's coarsen().mean() average the time coordinate, matching
    # exactly what both streams do per chunk (daily-mean label at 09Z).
    if time_coarsen_factor > 1:
        output_time = list(
            ds_ocean["time"]
            .coarsen(time=time_coarsen_factor, boundary="trim")
            .mean()
            .values
        )
    else:
        output_time = list(ds_ocean.time.values)

    logging.info(
        "Output time range: %s to %s (%d steps)",
        output_time[0],
        output_time[-1],
        len(output_time),
    )

    # --- Build template ---
    logging.info("Generating template")
    template, invariant_ds, ocean_nn_fill_map, atmo_nn_fill_map = _make_template(
        ds_ocean,
        ds_atmo,
        args.output_grid,
        vert_indices,
        time_coarsen_factor,
        output_time,
    )

    logging.info("Template generated. Starting pipeline.")
    output_store = _make_zarr_store(args.output_path, read_only=False)
    logging.info(
        "Pipeline options: %s",
        PipelineOptions(pipeline_args).get_all_options(),
    )

    # Each atmosphere chunk holds the 3-hourly timesteps covering one
    # ocean chunk (2 per 6-hourly ocean timestep).
    atmo_process_chunks = {"time": 2 * args.process_time_chunksize}

    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        # Stream 1: ocean (6-hourly MOM6)
        (
            p
            | "ocean_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_ocean, chunks=process_chunks)
            | beam.MapTuple(
                process_ocean_chunk,
                output_grid=args.output_grid,
                vertical_coarsening_indices=vert_indices,
                time_coarsen_factor=time_coarsen_factor,
                invariant_ds=invariant_ds,
                nn_fill_map=ocean_nn_fill_map,
            )
            | "ocean_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "ocean_to_zarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )

        # Stream 2: atmosphere (3-hourly FV3)
        (
            p
            | "atmo_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_atmo, chunks=atmo_process_chunks)
            | beam.MapTuple(
                process_atmo_chunk,
                output_grid=args.output_grid,
                time_coarsen_factor=time_coarsen_factor,
                invariant_ds=invariant_ds,
                nn_fill_map=atmo_nn_fill_map,
            )
            | "atmo_ConsolidateChunks" >> xbeam.ConsolidateChunks(output_shards)
            | "atmo_to_zarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    main()
