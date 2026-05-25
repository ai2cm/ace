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
    "prate_ave": "PRATEsfc",
    "frozr": "total_frozen_precipitation_rate",
}

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
    mask_2d = coarsened["mask_0"].copy()
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
    coarsened["mask_2d"] = mask_2d
    land_frac = (1.0 - mask_2d).astype(np.float32)
    land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
    coarsened["land_fraction"] = land_frac
    ssf = mask_2d.copy()
    ssf.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
    coarsened["sea_surface_fraction"] = ssf

    # Drop stray coordinates
    keep_coords = {"time", "lat", "lon"}
    cleaned = {}
    for name, da in coarsened.items():
        stray = [c for c in da.coords if c not in keep_coords and c not in da.dims]
        cleaned[name] = da.drop_vars(stray) if stray else da

    return xr.Dataset(cleaned)


def _split_3d_to_levels(
    ds: xr.Dataset,
    vars_3d: Sequence[str],
    vdim: str,
    mask_3d: xr.DataArray,
) -> xr.Dataset:
    """Split 3-D variables into per-level 2-D variables (no coarsening)."""
    n_levels = ds.sizes[vdim]
    new_vars: dict[str, xr.DataArray] = {}
    depths = ds[vdim].values
    keep_coords = {"time", "lat", "lon"}

    for var in vars_3d:
        if var not in ds:
            continue
        for i in range(n_levels):
            da = ds[var].isel({vdim: i}, drop=True)
            da = da.drop_vars([c for c in da.coords if c not in keep_coords])
            long_name = ds[var].attrs.get("long_name", var)
            da.attrs["long_name"] = f"{long_name} level-{i}"
            new_vars[f"{var}_{i}"] = da

    for i in range(n_levels):
        mask_i = mask_3d.isel({vdim: i}, drop=True).astype(np.float32)
        mask_i = mask_i.drop_vars([c for c in mask_i.coords if c not in keep_coords])
        mask_i.attrs = {"long_name": f"ocean mask level-{i}"}
        new_vars[f"mask_{i}"] = mask_i

    for i, depth in enumerate(depths):
        new_vars[f"idepth_{i}"] = xr.DataArray(
            float(depth),
            attrs={"units": "meters", "long_name": f"Depth at level-{i}"},
        )

    return xr.Dataset(new_vars)


# ---------------------------------------------------------------------------
# Nearest-neighbour fill for residual coastal NaN
# ---------------------------------------------------------------------------


def _nn_fill_2d_vars(ds: xr.Dataset, mask_2d: np.ndarray) -> xr.Dataset:
    """Fill NaN in 2-D ocean variables using nearest valid neighbour.

    Only applies to time-varying 2-D variables (not per-level 3-D splits).
    """
    from scipy.ndimage import distance_transform_edt

    level_prefixes = tuple(f"{v}_" for v in VARS_3D)
    ocean = mask_2d > 0

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
        logging.info("NN-filling %d ocean cells for '%s'", n_fill, name)

        valid = ~np.isnan(sample)
        _, nn_idx = distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        src_rows = nn_idx[0][need_fill]
        src_cols = nn_idx[1][need_fill]

        nlat, nlon = sample.shape
        fill_flat = np.ravel_multi_index(np.where(need_fill), (nlat, nlon))
        src_flat = np.ravel_multi_index((src_rows, src_cols), (nlat, nlon))

        # Vectorized fill over all timesteps
        data = v.values  # (time, lat, lon) — already in memory for Beam
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

    The UFS zarr stores may use different calendars (Julian for MOM6,
    possibly standard/proleptic_gregorian for FV3).  This function
    inspects the actual time values in the dataset and converts
    accordingly.
    """
    import cftime

    if isinstance(reference_time, cftime.datetime):
        # Match the exact cftime subclass used in the dataset
        cf_cls = type(reference_time)
        if isinstance(dt, cf_cls):
            return dt
        if isinstance(dt, (pd.Timestamp, datetime.datetime)):
            return cf_cls(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        if isinstance(dt, cftime.datetime):
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
) -> xr.Dataset:
    """Process one time-chunk of ocean + atmosphere data.

    This is the core function that each Beam worker executes.  It receives
    loaded (in-memory) xarray Datasets for a small number of timesteps
    and returns the fully processed output Dataset.
    """
    xr.set_options(keep_attrs=True)

    # Detect vertical dim
    vdim = "z_l" if "z_l" in ds_ocean.dims else "zl"

    # Handle duplicate vertical dims: move ho from zl to z_l if needed
    if "zl" in ds_ocean.dims and vdim != "zl":
        if "ho" in ds_ocean and "zl" in ds_ocean["ho"].dims:
            ho_on_vdim = ds_ocean["ho"].rename({"zl": vdim})
            ho_on_vdim = ho_on_vdim.assign_coords({vdim: ds_ocean[vdim].values})
            ds_ocean = ds_ocean.drop_vars("ho")
            ds_ocean["ho"] = ho_on_vdim
        ds_ocean = ds_ocean.drop_dims("zl", errors="ignore")

    # Drop unneeded variables
    ds_ocean = ds_ocean.drop_vars(
        [v for v in OCEAN_DROP if v in ds_ocean], errors="ignore"
    )
    ds_ocean = ds_ocean.drop_vars("landsea_mask", errors="ignore")

    # Rename ocean variables
    rename_map = {k: v for k, v in OCEAN_RENAME.items() if k in ds_ocean}
    ds_ocean = ds_ocean.rename(rename_map)
    stress_map = {k: v for k, v in STRESS_RENAME.items() if k in ds_ocean}
    ds_ocean = ds_ocean.rename(stress_map)

    # Separate ho for independent regridding
    ho_ds = None
    if "ho" in ds_ocean and vertical_coarsening_indices:
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
    ref_var = next(
        (v for v in ("thetao", "so") if v in ds_ocean and vdim in ds_ocean[v].dims),
        None,
    )
    if ref_var is not None:
        ref_slice = ds_ocean[ref_var].isel(time=0)
        mask_3d = (~np.isnan(ref_slice.values)).astype(np.float32)
        mask_3d = xr.DataArray(
            mask_3d,
            dims=ref_slice.dims,
            coords=ref_slice.coords,
            attrs={"long_name": "ocean mask from regridded NaN pattern"},
        )
    else:
        raise ValueError("Cannot build mask: no 3-D ocean variable found")

    mask_2d = mask_3d.isel({vdim: 0}).astype(np.float32)
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}

    # --- Vertical coarsening ---
    vars_3d_present = [v for v in VARS_3D if v in ds_ocean]

    if vertical_coarsening_indices:
        if ho_ds is None or "ho" not in ho_ds:
            raise ValueError("'ho' is required for thickness-weighted coarsening")
        ds_ocean["ho"] = ho_ds["ho"]
        indices_as_tuples = [tuple(pair) for pair in vertical_coarsening_indices]
        ds_levels = _compute_ocean_vertical_coarsening(
            ds_ocean,
            vars_3d_present,
            indices_as_tuples,
            vdim,
        )
    else:
        ds_ocean = ds_ocean.drop_vars("ho", errors="ignore")
        ds_levels = _split_3d_to_levels(
            ds_ocean,
            vars_3d_present,
            vdim,
            mask_3d,
        )

    # Collect 2-D ocean variables
    ocean_2d_names = [
        n
        for n in ds_ocean.data_vars
        if vdim not in ds_ocean[n].dims and n not in vars_3d_present and n != "ho"
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
        ds_levels["ssu"] = ds_levels["uo_0"].copy()
        ds_levels["ssu"].attrs = {
            "long_name": "Sea surface x-velocity",
            "units": "m/s",
        }
    if "vo_0" in ds_levels:
        ds_levels["ssv"] = ds_levels["vo_0"].copy()
        ds_levels["ssv"].attrs = {
            "long_name": "Sea surface y-velocity",
            "units": "m/s",
        }

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
        ds_ocean_2d["tauuo"] = ds_ocean_2d["eastward_surface_wind_stress"].copy()
        ds_ocean_2d["tauuo"].attrs = {
            "long_name": "Surface Downward X Stress",
            "units": "N/m2",
        }
    if "northward_surface_wind_stress" in ds_ocean_2d:
        ds_ocean_2d["tauvo"] = ds_ocean_2d["northward_surface_wind_stress"].copy()
        ds_ocean_2d["tauvo"].attrs = {
            "long_name": "Surface Downward Y Stress",
            "units": "N/m2",
        }

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

    # Restrict to common times (string-based to handle cross-calendar types)
    atmo_time_strs = {str(t) for t in ds_atmo_6h.time.values}
    ocean_mask = [str(t) in atmo_time_strs for t in ds_ocean.time.values]
    atmo_mask = [
        str(t) in {str(t) for t in ds_ocean.time.values[ocean_mask]}
        for t in ds_atmo_6h.time.values
    ]
    ds_atmo_6h = ds_atmo_6h.isel(time=atmo_mask)
    ds_ocean_2d = ds_ocean_2d.isel(time=ocean_mask)
    ds_levels = ds_levels.isel(time=ocean_mask)

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

    # Extract forcing and ice variables
    forcing_names = [k for k in ATMO_FORCING_VARS if k in ds_atmo_6h]
    ds_forcing = ds_atmo_6h[forcing_names].rename(
        {k: ATMO_FORCING_VARS[k] for k in forcing_names}
    )
    ice_names = [k for k in ICE_VARS if k in ds_atmo_6h]
    ds_ice = ds_atmo_6h[ice_names].rename({k: ICE_VARS[k] for k in ice_names})

    # --- Merge ---
    keep_coords = {"time", "lat", "lon"}
    to_merge = [ds_ocean_2d, ds_levels, ds_forcing, ds_ice]
    if not vertical_coarsening_indices:
        land_frac = (1.0 - mask_2d).astype(np.float32)
        land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
        ssf = mask_2d.copy()
        ssf.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
        to_merge.append(
            xr.Dataset(
                {
                    "mask_2d": mask_2d,
                    "land_fraction": land_frac,
                    "sea_surface_fraction": ssf,
                }
            )
        )

    cleaned = []
    for d in to_merge:
        stray = [c for c in d.coords if c not in keep_coords and c not in d.dims]
        cleaned.append(d.drop_vars(stray) if stray else d)
    ds = xr.merge(cleaned, join="inner")

    if "land_fraction" not in ds:
        land_frac = (1.0 - mask_2d).astype(np.float32)
        land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
        ds["land_fraction"] = land_frac

    # --- Insert NaN on land ---
    skip_mask = {"land_fraction", "sea_surface_fraction"}
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

    # Derived post-masking variables
    if "HI" in ds:
        ds["sea_ice_volume"] = ds["HI"].copy()
        ds["sea_ice_volume"].attrs = {
            "long_name": "Sea Ice Volume Per Area",
            "units": "m",
        }
    if "hfds" in ds and "sea_surface_fraction" in ds:
        ds["hfds_total_area"] = ds["hfds"] * ds["sea_surface_fraction"]
        ds["hfds_total_area"].attrs = {
            "long_name": "heat flux into sea water scaled by sea surface fraction",
            "units": "W/m2",
        }

    # --- NN fill ---
    mask_2d_arr = ds["mask_2d"].values if "mask_2d" in ds else mask_2d.values
    ds = _nn_fill_2d_vars(ds, mask_2d_arr)

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
    # the same dimensions.  Drop scalar variables (idepth_*) — they
    # are written once during template creation.  Broadcast 2-D
    # time-invariant fields (deptho, mask_*, land_fraction) to have
    # a time dim so ConsolidateChunks can combine them.
    scalars_to_drop = [
        name
        for name in ds.data_vars
        if ds[name].dims == ()
        or ("lat" not in ds[name].dims and "lon" not in ds[name].dims)
    ]
    if scalars_to_drop:
        ds = ds.drop_vars(scalars_to_drop)

    for name in list(ds.data_vars):
        if "time" not in ds[name].dims and "lat" in ds[name].dims:
            ds[name] = ds[name].expand_dims("time", axis=0)

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

    ds_atmo_6h = ds_atmo.resample(time="6h", closed="right", label="right").mean()

    # Match ocean and atmo times — they may be different cftime subclasses,
    # so compare by converting to strings.
    atmo_strs = [str(t) for t in ds_atmo_6h.time.values]
    ocean_strs = [str(t) for t in ocean_times.values]
    common_strs = set(atmo_strs) & set(ocean_strs)

    if not common_strs:
        raise ValueError(
            f"No overlapping times between averaged atmosphere and ocean. "
            f"Atmo 6h times: {atmo_strs[:5]}..., "
            f"Ocean times: {ocean_strs[:5]}..."
        )

    # Select by position using the string-matched indices
    mask = [str(t) in common_strs for t in ds_atmo_6h.time.values]
    return ds_atmo_6h.isel(time=mask)


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
    output_grid=DEFAULT_OUTPUT_GRID,
    vertical_coarsening_indices=None,
    time_coarsen_factor=1,
):
    """Beam-compatible processing function: (key, ds) → (new_key, output_ds).

    Called by ``beam.MapTuple`` for each time-chunk of ocean data.
    Loads the corresponding atmosphere data for the same time range.
    """
    if vertical_coarsening_indices is None:
        vertical_coarsening_indices = DEFAULT_VERTICAL_COARSENING_INDICES

    logging.info("Processing ocean chunk at key=%s", key)

    # Load the corresponding atmosphere time range.
    # Ocean and atmo may use different cftime subclasses, so convert.
    ocean_start = ds_ocean_chunk.time.values[0]
    ocean_end = ds_ocean_chunk.time.values[-1]
    atmo_ref = ds_atmo.time.values[0]

    atmo_start = _match_time_type(ocean_start, atmo_ref) - datetime.timedelta(hours=6)
    atmo_end = _match_time_type(ocean_end, atmo_ref) + datetime.timedelta(hours=3)
    ds_atmo_chunk = ds_atmo.sel(time=slice(atmo_start, atmo_end)).load()

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
    source_grid: xr.Dataset,
) -> xr.Dataset:
    """Extract time-invariant fields (idepth_*, deptho) from the ocean data.

    These are written once to the zarr store during template creation,
    not streamed through the Beam pipeline.
    """
    vdim = "z_l" if "z_l" in ds_ocean.dims else "zl"
    invariant = {}

    # idepth scalars from vertical coarsening config
    if vertical_coarsening_indices:
        depths = ds_ocean[vdim].values
        for i, (start, end) in enumerate(vertical_coarsening_indices):
            level_depths = depths[start:end]
            invariant[f"idepth_{i}"] = xr.DataArray(
                float(np.mean(level_depths)),
                attrs={"units": "meters", "long_name": f"Depth at level-{i}"},
            )
    else:
        depths = ds_ocean[vdim].values
        for i, depth in enumerate(depths):
            invariant[f"idepth_{i}"] = xr.DataArray(
                float(depth),
                attrs={"units": "meters", "long_name": f"Depth at level-{i}"},
            )

    return xr.Dataset(invariant)


def _make_template(
    ds_ocean: xr.Dataset,
    ds_atmo: xr.Dataset,
    output_grid: str,
    vertical_coarsening_indices: Sequence[Sequence[int]],
    time_coarsen_factor: int,
    output_time: list,
) -> xr.Dataset:
    """Eagerly process one ocean timestep to build the output zarr template."""
    logging.info("Building template from first timestep")

    # How many ocean timesteps form one output timestep
    ocean_per_output = max(1, time_coarsen_factor)

    # Load a small ocean chunk
    ds_ocean_small = ds_ocean.isel(time=slice(0, ocean_per_output)).load()

    # Load corresponding atmosphere (convert times to match atmo calendar)
    ocean_start = ds_ocean_small.time.values[0]
    ocean_end = ds_ocean_small.time.values[-1]
    atmo_ref = ds_atmo.time.values[0]
    atmo_start = _match_time_type(ocean_start, atmo_ref) - datetime.timedelta(hours=6)
    atmo_end = _match_time_type(ocean_end, atmo_ref) + datetime.timedelta(hours=3)
    ds_atmo_small = ds_atmo.sel(time=slice(atmo_start, atmo_end)).load()

    src_ocean, src_atmo = _get_source_grids(ds_ocean_small, ds_atmo_small)

    processed = _process_chunk(
        ds_ocean_small,
        ds_atmo_small,
        output_grid,
        vertical_coarsening_indices,
        time_coarsen_factor,
        src_ocean,
        src_atmo,
    )

    processed = processed.drop_encoding()

    # Separate invariant fields (scalars like idepth_*, and 2-D fields
    # like deptho/mask_*/land_fraction) that were dropped by _process_chunk.
    # We need to re-derive them for the template.  Re-run the coarsening
    # on the same small chunk but extract invariant fields before cleanup.
    # For simplicity, just add them back from a dedicated helper.
    invariant = _extract_invariant_fields(
        ds_ocean_small,
        output_grid,
        vertical_coarsening_indices,
        src_ocean,
    )
    invariant = invariant.drop_encoding()

    # Squeeze out the single-timestep time dim, then re-expand with the
    # full output time coordinate (same pattern as ERA5 pipeline).
    processed = processed.squeeze("time", drop=True)
    template = xbeam.make_template(processed)
    template = template.expand_dims(dim={"time": output_time}, axis=0)

    # Add invariant fields to template (they are written once, not
    # streamed through the Beam pipeline)
    for name in invariant.data_vars:
        if name not in template:
            template[name] = invariant[name]

    return template


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
    print(pipeline_args)

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
    atmo_load_vars = list(ATMO_FORCING_VARS.keys()) + list(ICE_VARS.keys())
    ds_atmo = open_atmo(atmo_load_vars, atmo_start, atmo_end_padded)
    logging.info("Atmo dataset: %s", dict(ds_atmo.sizes))

    # --- Output time coordinate ---
    # Derive from the actual ocean times after coarsening so the cftime
    # type and averaged values match what _process_chunk produces.
    ocean_times = ds_ocean.time.values
    if time_coarsen_factor > 1:
        # coarsen().mean() on cftime averages the timestamps, so we
        # replicate that here to get the exact output times.
        n_out = len(ocean_times) // time_coarsen_factor
        output_time = []
        for i in range(n_out):
            group = ocean_times[i * time_coarsen_factor : (i + 1) * time_coarsen_factor]
            # Average cftime by converting to seconds offset from first time
            ref = group[0]
            offsets_s = [(t - ref).total_seconds() for t in group]
            mean_offset = sum(offsets_s) / len(offsets_s)
            output_time.append(ref + datetime.timedelta(seconds=mean_offset))
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
    template = _make_template(
        ds_ocean,
        ds_atmo,
        args.output_grid,
        vert_indices,
        time_coarsen_factor,
        output_time,
    )

    logging.info("Template generated. Starting pipeline.")
    output_store = _make_zarr_store(args.output_path, read_only=False)
    print(PipelineOptions(pipeline_args).get_all_options())

    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | xbeam.DatasetToChunks(ds_ocean, chunks=process_chunks)
            | beam.MapTuple(
                process_ocean_chunk,
                ds_atmo=ds_atmo,
                output_grid=args.output_grid,
                vertical_coarsening_indices=vert_indices,
                time_coarsen_factor=time_coarsen_factor,
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
