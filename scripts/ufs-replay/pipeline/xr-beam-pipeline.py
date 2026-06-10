"""
xarray-beam pipeline for the UFS GEFSv13 replay ocean (MOM6) + atmosphere
(FV3) dataset.  Produces a training-ready zarr store for SamudrACE-type
models, matching the runner/infrastructure pattern of ``scripts/era5/``.

Pipeline steps (applied per time-chunk by each Beam worker):

  1. Read MOM6 ocean variables (3-D + surface + layer thickness ``ho``).
  2. Read FV3 atmosphere forcing and sea-ice variables.
  3. Average 3-hourly FV3 atmosphere to 6-hourly ocean cadence.
  4. Regrid ocean and atmosphere to a Gaussian grid via xESMF.
  5. Apply thickness-weighted vertical coarsening (``ho``-weighted).
  6. Split 3-D fields into per-level 2-D variables.
  7. Derive additional variables (SST, ssu/ssv, deptho, wfo, hfds, etc.).
  8. Insert NaN on land, nearest-neighbour fill residual coastal NaN.
  9. Optionally coarsen in time (e.g. 6-hourly → daily).

Data sources::

    gs://noaa-ufs-gefsv13replay/ufs-hr1/{resolution}/{freq}/zarr/

Usage::

    python xr-beam-pipeline.py <output_path> <start_time> <end_time> \\
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

    Returns a flat Dataset with per-level 2-D fields, masks, and depth
    scalars — ready for direct merging into the output.
    """
    depths = ds[vdim].values

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

        # Per-level mask from coarsened data validity
        ref = None
        for vn in vars_3d:
            key = f"{vn}_{i}"
            if key in coarsened:
                ref = coarsened[key]
                break
        if ref is not None:
            mask_i = ref.isel(time=0).notnull().astype(np.float32)
        else:
            ho_total = ho_slice.sum(vdim, skipna=True)
            mask_i = (ho_total.isel(time=0) > 0).astype(np.float32)
        mask_i.attrs = {"long_name": f"ocean mask level-{i}"}
        coarsened[f"mask_{i}"] = mask_i

        level_depths = depths[start:end]
        coarsened[f"idepth_{i}"] = xr.DataArray(
            float(np.mean(level_depths)),
            attrs={"units": "meters", "long_name": f"Depth at level-{i}"},
        )

    # Pass through 2-D variables
    for name in ds.data_vars:
        if name in vars_3d or name == thickness_name:
            continue
        if vdim not in ds[name].dims:
            coarsened[name] = ds[name]

    # Surface mask + related fields
    mask_2d = coarsened["mask_0"].astype(np.float32)
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
    coarsened["mask_2d"] = mask_2d
    coarsened.update(_build_land_sea_fractions(mask_2d))

    return xr.Dataset(coarsened)


# ---------------------------------------------------------------------------
# Mask helpers (shared by _process_chunk and _extract_invariant_fields)
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


def _build_land_sea_fractions(
    mask_2d: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Derive land_fraction and sea_surface_fraction from mask_2d."""
    land_frac = (1.0 - mask_2d).astype(np.float32)
    land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
    ssf = mask_2d.astype(np.float32)
    ssf.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
    return {"land_fraction": land_frac, "sea_surface_fraction": ssf}


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


# ---------------------------------------------------------------------------
# Per-chunk processing (called by each Beam worker)
# ---------------------------------------------------------------------------


def _process_chunk(
    ds_ocean: xr.Dataset,
    ds_atmo: xr.Dataset,
    output_grid: str,
    vertical_coarsening_indices: Sequence[Sequence[int]],
    time_coarsen_factor: int,
    source_grid_ocean: xr.Dataset,
    source_grid_atmo: xr.Dataset,
    nn_fill_map: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> xr.Dataset:
    """Process one time-chunk of ocean + atmosphere data.

    This is the core function that each Beam worker executes.  It receives
    loaded (in-memory) xarray Datasets for a small number of timesteps
    and returns the fully processed output Dataset.
    """
    xr.set_options(keep_attrs=True)

    # Separate ho for independent regridding (thickness-weighted
    # coarsening needs ho regridded without NaN-masking influence
    # from tracer fields that have deeper NaN patterns).
    ho_ds = None
    if "ho" in ds_ocean:
        ho_ds = ds_ocean[["ho"]]
        ds_ocean = ds_ocean.drop_vars("ho")

    # --- Horizontal regridding ---
    if output_grid:
        ds_ocean = _regrid_dataset(
            ds_ocean,
            output_grid,
            source_grid_ocean,
            skipna=True,
            na_thres=1.0,
        )
        if ho_ds is not None:
            ho_ds = _regrid_dataset(
                ho_ds,
                output_grid,
                source_grid_ocean,
                skipna=True,
                na_thres=1.0,
            )
            ho_ds = ho_ds.assign_coords(
                lat=ds_ocean.lat.values, lon=ds_ocean.lon.values
            )

    # Rebuild masks from regridded NaN pattern
    mask_3d, mask_2d = _build_3d_mask(ds_ocean, VDIM, time_idx=0)

    # --- Vertical coarsening ---
    vars_3d_present = [v for v in VARS_3D if v in ds_ocean]

    if ho_ds is None or "ho" not in ho_ds:
        raise ValueError("'ho' is required for thickness-weighted coarsening")
    ds_ocean["ho"] = ho_ds["ho"]
    indices_as_tuples = [tuple(pair) for pair in vertical_coarsening_indices]
    ds_levels = _compute_ocean_vertical_coarsening(
        ds_ocean,
        vars_3d_present,
        indices_as_tuples,
        VDIM,
    )

    # Collect 2-D ocean variables
    ocean_2d_names = [
        n
        for n in ds_ocean.data_vars
        if VDIM not in ds_ocean[n].dims and n not in vars_3d_present and n != "ho"
    ]
    ds_ocean_2d = ds_ocean[ocean_2d_names]

    # --- Derived variables ---
    # SST in Kelvin
    if "thetao_0" in ds_levels:
        sst_K = ds_levels["thetao_0"] + 273.15
        sst_K.attrs = {"long_name": "Sea surface temperature", "units": "K"}
        ds_levels["sst"] = sst_K
    if "zos" in ds_ocean_2d:
        ds_ocean_2d["zos"].attrs.setdefault("long_name", "Sea Surface Height")

    # Surface velocity aliases
    if "uo_0" in ds_levels:
        ssu = ds_levels["uo_0"]
        ssu.attrs = {"long_name": "Sea surface x-velocity", "units": "m/s"}
        ds_levels["ssu"] = ssu
    if "vo_0" in ds_levels:
        ssv = ds_levels["vo_0"]
        ssv.attrs = {"long_name": "Sea surface y-velocity", "units": "m/s"}
        ds_levels["ssv"] = ssv

    # deptho from MOM6 "depth" (time-invariant)
    if "depth" in ds_ocean_2d:
        ds_ocean_2d["deptho"] = ds_ocean_2d["depth"]
        ds_ocean_2d["deptho"].attrs = {
            "long_name": "Sea Floor Depth Below Geoid",
            "units": "m",
        }
        ds_ocean_2d = ds_ocean_2d.drop_vars("depth")

    # Stress aliases
    if "eastward_surface_wind_stress" in ds_ocean_2d:
        tauuo = ds_ocean_2d["eastward_surface_wind_stress"]
        tauuo.attrs = {"long_name": "Surface Downward X Stress", "units": "N/m2"}
        ds_ocean_2d["tauuo"] = tauuo
    if "northward_surface_wind_stress" in ds_ocean_2d:
        tauvo = ds_ocean_2d["northward_surface_wind_stress"]
        tauvo.attrs = {"long_name": "Surface Downward Y Stress", "units": "N/m2"}
        ds_ocean_2d["tauvo"] = tauvo

    # wfo: water flux = evap + lprec + fprec + lrunoff
    if all(v in ds_ocean_2d for v in WFO_COMPONENTS):
        wfo = sum(ds_ocean_2d[c] for c in WFO_COMPONENTS)
        wfo.attrs = {
            "long_name": "Water Flux Into Sea Water",
            "units": "kg/(m2 s)",
        }
        ds_ocean_2d["wfo"] = wfo

    # hfds: net surface heat flux
    if all(v in ds_ocean_2d for v in HFDS_COMPONENTS):
        hfds = sum(ds_ocean_2d[c] for c in HFDS_COMPONENTS)
        hfds.attrs = {
            "long_name": "Downward Heat Flux at Sea Water Surface",
            "units": "W/m2",
        }
        ds_ocean_2d["hfds"] = hfds

    # Drop raw flux components
    ds_ocean_2d = ds_ocean_2d.drop_vars(
        [v for v in WFO_COMPONENTS + HFDS_COMPONENTS if v in ds_ocean_2d],
        errors="ignore",
    )

    # --- Atmosphere processing ---
    # Average 3-hourly → 6-hourly, then regrid
    ocean_times = ds_ocean.time
    ds_atmo_6h = _average_atmo_chunk(ds_atmo, ocean_times)

    if output_grid:
        ds_atmo_6h = _regrid_dataset(
            ds_atmo_6h,
            output_grid,
            source_grid_atmo,
            skipna=True,
            na_thres=1.0,
        )
        ds_atmo_6h = ds_atmo_6h.assign_coords(
            lat=ds_ocean.lat.values, lon=ds_ocean.lon.values
        )

    # Restrict to common times
    common_times = sorted(set(ds_ocean.time.values) & set(ds_atmo_6h.time.values))
    ds_atmo_6h = ds_atmo_6h.sel(time=common_times)
    ds_ocean_2d = ds_ocean_2d.sel(time=common_times)
    ds_levels = ds_levels.sel(time=common_times)

    # Time coarsening
    if time_coarsen_factor > 1:
        n_times = ds_ocean_2d.sizes["time"]
        if n_times < time_coarsen_factor:
            raise ValueError(
                f"Chunk has {n_times} timesteps after time intersection, "
                f"but time_coarsen_factor={time_coarsen_factor}. "
                f"Ensure process_time_chunksize is a multiple of "
                f"time_coarsen_factor and that the time range is aligned."
            )
        ds_atmo_6h = ds_atmo_6h.coarsen(
            time=time_coarsen_factor, boundary="trim"
        ).mean()
        ds_ocean_2d = ds_ocean_2d.coarsen(
            time=time_coarsen_factor, boundary="trim"
        ).mean()
        ds_levels = ds_levels.coarsen(time=time_coarsen_factor, boundary="trim").mean()
        # Snap daily-mean time labels to 12Z (the natural midpoint of a day).
        # coarsen().mean() averages the timestamps, giving 09Z for (00,06,12,18).
        snapped = []
        for t in ds_ocean_2d.time.values:
            if isinstance(t, _cftime.datetime):
                snapped.append(type(t)(t.year, t.month, t.day, 12, 0, 0))
            else:
                snapped.append(t)
        ds_ocean_2d = ds_ocean_2d.assign_coords(time=snapped)
        ds_levels = ds_levels.assign_coords(time=snapped)
        ds_atmo_6h = ds_atmo_6h.assign_coords(time=snapped)

    # Extract forcing and ice variables
    forcing_names = [k for k in ATMO_FORCING_VARS if k in ds_atmo_6h]
    ds_forcing = ds_atmo_6h[forcing_names].rename(
        {k: ATMO_FORCING_VARS[k] for k in forcing_names}
    )
    ice_names = [k for k in ICE_VARS if k in ds_atmo_6h]
    ds_ice = ds_atmo_6h[ice_names].rename({k: ICE_VARS[k] for k in ice_names})

    # Derived atmo variables (already have their final names)
    derived_atmo_names = [
        v for v in ["total_frozen_precipitation_rate"] if v in ds_atmo_6h
    ]
    ds_derived_atmo = ds_atmo_6h[derived_atmo_names] if derived_atmo_names else None

    # --- Merge ---
    keep_coords = {"time", "lat", "lon"}
    to_merge = [ds_ocean_2d, ds_levels, ds_forcing, ds_ice]
    if ds_derived_atmo is not None:
        to_merge.append(ds_derived_atmo)

    cleaned = []
    for d in to_merge:
        stray = [c for c in d.coords if c not in keep_coords and c not in d.dims]
        cleaned.append(d.drop_vars(stray) if stray else d)
    ds = xr.merge(cleaned, join="inner")

    if "land_fraction" not in ds:
        fractions = _build_land_sea_fractions(mask_2d)
        ds["land_fraction"] = fractions["land_fraction"]

    # --- Insert NaN on land ---
    # Atmospheric forcing variables are valid globally (FV3 is a global
    # model) and must NOT be masked — downstream training configs rely
    # on having values everywhere including over land.
    atmo_skip = set(ATMO_FORCING_VARS.values())
    atmo_skip.add("total_frozen_precipitation_rate")
    skip_mask = {"land_fraction", "sea_surface_fraction"} | atmo_skip
    for name in list(ds.data_vars):
        if name.startswith("mask_") or name.startswith("idepth_"):
            continue
        if name in skip_mask:
            continue
        v = ds[name]
        if "lat" not in v.dims or "lon" not in v.dims:
            continue
        if "time" in v.dims:
            level_match = name.rsplit("_", 1)
            if len(level_match) == 2 and level_match[1].isdigit():
                level_idx = int(level_match[1])
                mask_name = f"mask_{level_idx}"
                if mask_name in ds:
                    ds[name] = v.where(ds[mask_name] > 0)
                    continue
        ds[name] = v.where(ds["mask_2d"] > 0)

    # Sea ice special handling
    if "ocean_sea_ice_fraction" in ds:
        ds["ocean_sea_ice_fraction"] = ds["ocean_sea_ice_fraction"].where(
            ds["mask_2d"] > 0, np.nan
        )
    if "HI" in ds:
        ds["HI"] = ds["HI"].where(ds["mask_2d"] > 0, np.nan)
        if "ocean_sea_ice_fraction" in ds:
            ds["HI"] = ds["HI"].where(ds["ocean_sea_ice_fraction"] > 0, 0.0)
        # Re-apply NaN on land — the ice-free zeroing above converts land
        # from NaN to 0.0 (because NaN > 0 is False), but ACE's output
        # masker expects NaN on land to match the target NaN pattern.
        ds["HI"] = ds["HI"].where(ds["mask_2d"] > 0, np.nan)

    # Derived post-masking variables
    if "HI" in ds:
        siv = ds["HI"]
        siv.attrs = {"long_name": "Sea Ice Volume Per Area", "units": "m"}
        ds["sea_ice_volume"] = siv
    if "hfds" in ds and "sea_surface_fraction" in ds:
        ds["hfds_total_area"] = ds["hfds"] * ds["sea_surface_fraction"]
        ds["hfds_total_area"].attrs = {
            "long_name": "heat flux into sea water scaled by sea surface fraction",
            "units": "W/m2",
        }

    # --- NN fill ---
    if nn_fill_map is not None:
        ds = _apply_nn_fill(ds, nn_fill_map)
    else:
        mask_2d_arr = ds["mask_2d"].values if "mask_2d" in ds else mask_2d.values
        nn_fill_map = _compute_nn_fill_indices(ds, mask_2d_arr)
        ds = _apply_nn_fill(ds, nn_fill_map)

    # --- Clean up ---
    # Drop raw MOM6 flux components that were only needed for deriving
    # wfo/hfds/deptho but leaked into the merged dataset.
    _raw_vars_to_drop = WFO_COMPONENTS + HFDS_COMPONENTS + ["depth"]
    ds = ds.drop_vars([v for v in _raw_vars_to_drop if v in ds], errors="ignore")

    keep_dims = {"time", "lat", "lon"}
    drop_dims = [d for d in ds.dims if d not in keep_dims]
    if drop_dims:
        ds = ds.drop_dims(drop_dims)
    ds = ds.reset_coords(drop=True)

    for name in ds.data_vars:
        if ds[name].dtype not in (np.float32, np.int32):
            ds[name] = ds[name].astype(np.float32)

    # xarray-beam's ConsolidateChunks requires all variables to share
    # the same dimensions.  Drop scalar and time-invariant spatial
    # variables — they are written once during template creation.
    TIME_INVARIANT_SPATIAL = {
        "mask_2d",
        "land_fraction",
        "sea_surface_fraction",
        "deptho",
    }
    TIME_INVARIANT_SPATIAL.update(
        name for name in ds.data_vars if name.startswith("mask_")
    )
    invariant_to_drop = [
        name
        for name in ds.data_vars
        if ds[name].dims == ()
        or ("lat" not in ds[name].dims and "lon" not in ds[name].dims)
        or name in TIME_INVARIANT_SPATIAL
    ]
    if invariant_to_drop:
        ds = ds.drop_vars(invariant_to_drop)

    return ds


def _average_atmo_chunk(
    ds_atmo: xr.Dataset,
    ocean_times: xr.DataArray,
) -> xr.Dataset:
    """Average 3-hourly atmosphere data to 6-hourly ocean cadence.

    Expects *ds_atmo* to already be sliced to the relevant time range
    (with ±buffer) and loaded into memory.
    """
    if ds_atmo.sizes["time"] == 0:
        raise ValueError(
            "Atmosphere dataset has 0 timesteps — check that the atmo zarr "
            "time range overlaps the ocean range and that calendars match."
        )

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

    ds_atmo_6h = ds_atmo.resample(time="6h", closed="right", label="right").mean()

    common_times = sorted(set(ds_atmo_6h.time.values) & set(ocean_times.values))

    if not common_times:
        raise ValueError(
            f"No overlapping times between averaged atmosphere and ocean. "
            f"Atmo 6h times: {ds_atmo_6h.time.values[:5]}..., "
            f"Ocean times: {ocean_times.values[:5]}..."
        )

    return ds_atmo_6h.sel(time=common_times)


# ---------------------------------------------------------------------------
# Beam-compatible process function
# ---------------------------------------------------------------------------

# Source grids are built lazily and cached per worker
_SOURCE_GRID_CACHE = {}


def _get_source_grids(ds_ocean: xr.Dataset, ds_atmo: xr.Dataset):
    """Build and cache source grid descriptors."""
    ocean_key = (len(ds_ocean["lat"]), len(ds_ocean["lon"]))
    atmo_key = (len(ds_atmo["lat"]), len(ds_atmo["lon"]))
    if ocean_key not in _SOURCE_GRID_CACHE:
        _SOURCE_GRID_CACHE[ocean_key] = _make_source_grid(ds_ocean)
    if atmo_key not in _SOURCE_GRID_CACHE:
        _SOURCE_GRID_CACHE[atmo_key] = _make_source_grid(ds_atmo)
    return _SOURCE_GRID_CACHE[ocean_key], _SOURCE_GRID_CACHE[atmo_key]


def process_ocean_chunk(
    key,
    ds_ocean_chunk,
    ds_atmo=None,
    atmo_isel_map=None,
    output_grid=DEFAULT_OUTPUT_GRID,
    vertical_coarsening_indices=None,
    time_coarsen_factor=1,
    nn_fill_map=None,
):
    """Beam-compatible processing function: (key, ds) → (new_key, output_ds).

    Called by ``beam.MapTuple`` for each time-chunk of ocean data.
    Loads the corresponding atmosphere data for the same time range.
    """
    if atmo_isel_map is None:
        atmo_isel_map = {}
    if vertical_coarsening_indices is None:
        vertical_coarsening_indices = DEFAULT_VERTICAL_COARSENING_INDICES

    logging.info("Processing ocean chunk at key=%s", key)

    # Load the corresponding atmosphere time range using pre-computed
    # integer index mapping (avoids cftime serialization/matching issues).
    ocean_time_offset = key.offsets["time"]
    if ocean_time_offset in atmo_isel_map:
        atmo_slice = atmo_isel_map[ocean_time_offset]
        ds_atmo_chunk = ds_atmo.isel(time=atmo_slice).load()
    else:
        logging.warning(
            "No atmo_isel_map entry for ocean offset=%d, falling back to "
            "time-based selection",
            ocean_time_offset,
        )
        ocean_start = ds_ocean_chunk.time.values[0]
        ocean_end = ds_ocean_chunk.time.values[-1]
        atmo_ref = ds_atmo.time.values[0]
        atmo_start = _match_time_type(ocean_start, atmo_ref) - datetime.timedelta(
            hours=6
        )
        atmo_end = _match_time_type(ocean_end, atmo_ref) + datetime.timedelta(hours=3)
        ds_atmo_chunk = ds_atmo.sel(time=slice(atmo_start, atmo_end)).load()

    logging.info(
        "Atmo chunk: %d timesteps for ocean offset=%d",
        ds_atmo_chunk.sizes["time"],
        ocean_time_offset,
    )

    # Load ocean chunk
    ds_ocean_chunk = ds_ocean_chunk.load()

    # Build source grids (cached per worker)
    src_ocean, src_atmo = _get_source_grids(ds_ocean_chunk, ds_atmo_chunk)

    output = _process_chunk(
        ds_ocean_chunk,
        ds_atmo_chunk,
        output_grid,
        vertical_coarsening_indices,
        time_coarsen_factor,
        src_ocean,
        src_atmo,
        nn_fill_map=nn_fill_map,
    )

    # Update key for output dimensions
    # Time coarsening changes the offset mapping
    if time_coarsen_factor > 1:
        output_time_offset = key.offsets["time"] // time_coarsen_factor
    else:
        output_time_offset = key.offsets["time"]

    new_key = key.replace(
        offsets={"time": output_time_offset},
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

    Returns scalar fields (idepth_*) and 2-D spatial fields (mask_*,
    land_fraction, sea_surface_fraction, deptho) without a time dimension.
    These are written once to the zarr store during template creation,
    not streamed through the Beam pipeline.
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

    # 2-D spatial invariant fields: derive from a single regridded
    # ocean timestep (mask, land_fraction, sea_surface_fraction, deptho).
    ds_ocean_1t = ds_ocean.isel(time=0).load()

    # Regrid to get the output-grid mask
    if output_grid:
        ds_ocean_1t = _regrid_dataset(
            ds_ocean_1t,
            output_grid,
            source_grid_ocean,
            skipna=True,
            na_thres=1.0,
        )

    # Build masks using shared helpers
    mask_3d, mask_2d = _build_3d_mask(ds_ocean_1t, VDIM)
    level_masks = _build_per_level_masks(mask_3d, VDIM, vertical_coarsening_indices)
    invariant.update(level_masks)
    invariant["mask_2d"] = mask_2d
    invariant.update(_build_land_sea_fractions(mask_2d))

    # deptho
    depth_name = "depth" if "depth" in ds_ocean_1t else "deptho"
    if depth_name in ds_ocean_1t:
        deptho = ds_ocean_1t[depth_name].astype(np.float32)
        deptho.attrs = {"long_name": "Sea Floor Depth Below Geoid", "units": "m"}
        invariant["deptho"] = deptho

    return xr.Dataset(invariant)


def _make_template(
    ds_ocean: xr.Dataset,
    ds_atmo: xr.Dataset,
    output_grid: str,
    vertical_coarsening_indices: Sequence[Sequence[int]],
    time_coarsen_factor: int,
    output_time: list,
) -> tuple[xr.Dataset, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Eagerly process one ocean timestep to build the output zarr template.

    Returns (template, nn_fill_map) where nn_fill_map contains precomputed
    nearest-neighbour fill indices to be reused by every worker.
    """
    logging.info("Building template from first timestep")

    # Extract scalar (idepth_*) and 2-D spatial (mask_*, land_fraction,
    # etc.) invariant fields — all without a time dimension.
    src_ocean = _make_source_grid(ds_ocean.isel(time=0).load())
    invariant = _extract_invariant_fields(
        ds_ocean,
        output_grid,
        vertical_coarsening_indices,
        src_ocean,
    )
    invariant = invariant.drop_encoding()

    # Process one chunk to get the time-varying variable schema.
    # _process_chunk drops invariant fields, which is what we want
    # for the template (they're added separately below).
    ocean_per_output = max(1, time_coarsen_factor)
    ds_ocean_small = ds_ocean.isel(time=slice(0, ocean_per_output)).load()
    ocean_start = ds_ocean_small.time.values[0]
    ocean_end = ds_ocean_small.time.values[-1]
    atmo_ref = ds_atmo.time.values[0]
    atmo_start = _match_time_type(ocean_start, atmo_ref) - datetime.timedelta(hours=6)
    atmo_end = _match_time_type(ocean_end, atmo_ref) + datetime.timedelta(hours=3)
    ds_atmo_small = ds_atmo.sel(time=slice(atmo_start, atmo_end)).load()
    src_atmo = _make_source_grid(ds_atmo_small)

    # First pass: process WITHOUT NN-fill so we can extract fill indices
    # from the un-filled data.  Then apply fill and continue.
    processed = _process_chunk(
        ds_ocean_small,
        ds_atmo_small,
        output_grid,
        vertical_coarsening_indices,
        time_coarsen_factor,
        src_ocean,
        src_atmo,
        nn_fill_map={},  # empty map → no fill applied, no fallback
    )
    processed = processed.drop_encoding()

    # Precompute NN-fill indices from the UN-FILLED template chunk.
    # The fill pattern depends only on the static ocean mask, so we
    # compute it once here and pass to every Beam worker.
    mask_2d_arr = (
        invariant["mask_2d"].values
        if "mask_2d" in invariant
        else processed["mask_2d"].values
    )
    nn_fill_map = _compute_nn_fill_indices(processed, mask_2d_arr)

    # Now apply NN fill to the template data
    processed = _apply_nn_fill(processed, nn_fill_map)

    # Squeeze out the single-timestep time dim, then re-expand with the
    # full output time coordinate (same pattern as ERA5 pipeline).
    processed = processed.squeeze("time", drop=True)
    template = xbeam.make_template(processed)
    template = template.expand_dims(dim={"time": output_time}, axis=0)

    # Add invariant fields to template (written once, no time dimension)
    for name in invariant.data_vars:
        if name not in template:
            template[name] = invariant[name]

    return template, nn_fill_map


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
    parser.add_argument(
        "--check_data_validity",
        action="store_true",
        help="Check for unexpected NaN values before processing.",
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
    logging.info("Ocean dataset: %s", dict(ds_ocean.sizes))

    # Atmosphere: need 3-hourly data spanning the full ocean range + buffer
    atmo_start = start_time - datetime.timedelta(hours=6)
    atmo_end_padded = end_time + datetime.timedelta(hours=3)
    atmo_load_vars = (
        list(ATMO_FORCING_VARS.keys())
        + list(ICE_VARS.keys())
        + FROZEN_PRECIP_ACCUM_VARS
    )
    ds_atmo = open_atmo(atmo_load_vars, atmo_start, atmo_end_padded)
    logging.info("Atmo dataset: %s", dict(ds_atmo.sizes))

    # Pre-compute integer index mapping from ocean chunk offset → atmo
    # time slice so workers can select by position.
    ocean_times_arr = ds_ocean.time.values
    atmo_times_arr = ds_atmo.time.values
    atmo_isel_map = {}  # ocean_time_offset → slice(start, end)
    pcs = args.process_time_chunksize
    for chunk_idx in range(len(ocean_times_arr) // pcs):
        ocean_offset = chunk_idx * pcs
        ocean_chunk_start = ocean_times_arr[ocean_offset]
        ocean_chunk_end = ocean_times_arr[ocean_offset + pcs - 1]
        # Atmo window: 6h before first ocean time to 3h after last
        a_start = ocean_chunk_start - datetime.timedelta(hours=6)
        a_end = ocean_chunk_end + datetime.timedelta(hours=3)
        idx_start = None
        idx_end = None
        for i, t in enumerate(atmo_times_arr):
            if t >= a_start:
                if idx_start is None:
                    idx_start = i
                if t <= a_end:
                    idx_end = i
        if idx_start is not None and idx_end is not None:
            atmo_isel_map[ocean_offset] = slice(idx_start, idx_end + 1)
        else:
            logging.warning(
                "Could not find atmo indices for ocean offset=%d "
                "(ocean %s to %s, atmo window %s to %s)",
                ocean_offset,
                ocean_chunk_start,
                ocean_chunk_end,
                a_start,
                a_end,
            )
    logging.info(
        "Built atmo_isel_map with %d entries for %d ocean chunks",
        len(atmo_isel_map),
        len(ocean_times_arr) // pcs,
    )

    # --- Output time coordinate ---
    # Snap daily-mean time labels to 12Z to match _process_chunk.
    ocean_times = ds_ocean.time.values
    if time_coarsen_factor > 1:
        n_out = len(ocean_times) // time_coarsen_factor
        output_time = []
        for i in range(n_out):
            group = ocean_times[i * time_coarsen_factor : (i + 1) * time_coarsen_factor]
            ref = group[0]
            if isinstance(ref, _cftime.datetime):
                output_time.append(type(ref)(ref.year, ref.month, ref.day, 12, 0, 0))
            else:
                output_time.append(ref)
    else:
        output_time = list(ocean_times)

    logging.info(
        "Output time range: %s to %s (%d steps)",
        output_time[0],
        output_time[-1],
        len(output_time),
    )

    # --- Build template ---
    logging.info("Generating template")
    template, nn_fill_map = _make_template(
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

    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | xbeam.DatasetToChunks(ds_ocean, chunks=process_chunks)
            | beam.MapTuple(
                process_ocean_chunk,
                ds_atmo=ds_atmo,
                atmo_isel_map=atmo_isel_map,
                output_grid=args.output_grid,
                vertical_coarsening_indices=vert_indices,
                time_coarsen_factor=time_coarsen_factor,
                nn_fill_map=nn_fill_map,
            )
            | xbeam.ConsolidateChunks(output_shards)
            | xbeam.ChunksToZarr(
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
