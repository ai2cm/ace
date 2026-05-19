"""
Preprocessing for the UFS GEFSv13 replay ocean (MOM6) + atmosphere (FV3)
dataset, producing a training-ready zarr store for SamudrACE-type models.

The GCP zarr stores (0.25-degree) cover the full ~30-year record (1994-2023).
This script reads from them, applies xESMF conservative regridding to a
Gaussian target grid (matching the ERA5 pipeline in ``scripts/era5/``), and
produces the training-ready output.

Pipeline steps:

  1. Load the 0.25-degree MOM6 ocean zarr (3-D + surface ocean variables).
  2. Load the 0.25-degree FV3 atmosphere zarr and extract surface forcing
     fields needed for uncoupled ocean training.
  3. Apply xESMF conservative horizontal regridding to a Gaussian target grid
     (default F90 = 180×360 ≈ 1-degree, same as ERA5 pipeline).
  4. Optionally apply thickness-weighted vertical coarsening.
  5. Split 3-D fields into per-level 2-D variables (``thetao_0 … thetao_N``).
  6. Create land/ocean masks, insert NaNs on land, add SST in Kelvin.
  7. Merge sea-ice variables from FV3 (``icec``, ``icetk``).
  8. Write a sharded zarr store.

Data lives at::

    gs://noaa-ufs-gefsv13replay/ufs-hr1/{resolution}/{freq}/zarr/

where resolution is ``0.25-degree`` (full record) or ``1.00-degree`` (subset)
and freq is ``06h-freq`` (ocean) or ``03h-freq`` (atmosphere).

Usage::

    python compute_ocean_dataset_ufs_replay.py --config <yaml> --output-store <path>
"""

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import click
import dacite
import numpy as np
import xarray as xr
import yaml

try:
    import xpartition  # noqa: F401
except ImportError:
    pass

from compute_dataset import ChunkingConfig, clear_compressors_encoding
from dask.diagnostics import ProgressBar

# ──────────────────────────────────────────────────────────────────────────────
# Variable mapping
# ──────────────────────────────────────────────────────────────────────────────

# MOM6 zarr variable names  →  SamudrACE standard names
_OCEAN_RENAME: dict[str, str] = {
    "temp": "thetao",
    "SSH": "zos",
}

# FV3 zarr variable names  →  SamudrACE forcing names
_ATMO_FORCING_VARS: dict[str, str] = {
    "dlwrf_ave": "DLWRFsfc",
    "dswrf_ave": "DSWRFsfc",
    "ulwrf_ave": "ULWRFsfc",
    "uswrf_ave": "USWRFsfc",
    "lhtfl_ave": "LHTFLsfc",
    "shtfl_ave": "SHTFLsfc",
    "prate_ave": "PRATEsfc",
}

# Sea-ice variables extracted from FV3
_ICE_VARS: dict[str, str] = {
    "icec": "ocean_sea_ice_fraction",
    "icetk": "HI",
}

# Wind stress comes from MOM6 ocean store (already ocean-received stress)
_STRESS_RENAME: dict[str, str] = {
    "taux": "eastward_surface_wind_stress",
    "tauy": "northward_surface_wind_stress",
}


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class RegridConfig:
    """Configuration for horizontal regridding via xESMF.

    Uses the same Gaussian grid convention as the ERA5 pipeline in
    ``scripts/era5/pipeline/xr-beam-pipeline.py``.

    Attributes:
        output_grid: Gaussian grid name. ``"F90"`` → 180×360 (~1°),
            ``"F22.5"`` → 45×90 (~4°), ``"F360"`` → 720×1440 (~0.25°).
            Set to ``""`` or ``None`` to skip regridding (input is already on
            the target grid).
        method: xESMF regridding method (default ``"conservative"``).
    """

    output_grid: str = "F90"
    method: str = "conservative"


@dataclasses.dataclass
class VerticalCoarsenConfig:
    """Thickness-weighted vertical coarsening from 75 MOM6 levels.

    Uses the same ``(start, end)`` index-range approach as the atmosphere
    vertical coarsening in ``compute_dataset.py``, with MOM6 layer thickness
    (``ho``) as the weight instead of pressure thickness.

    Attributes:
        vertical_coarsening_indices: List of ``[start, end]`` pairs (half-open)
            defining which contiguous native levels to average into each coarse
            output level.  For example ``[[0, 5], [5, 10]]`` produces 2 coarse
            levels from native levels 0-4 and 5-9.
            If empty, all levels are kept without coarsening.
    """

    vertical_coarsening_indices: list[list[int]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class DaskConfig:
    n_workers: int = 16
    use_gateway: bool = False
    cluster_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    _cluster: Any = dataclasses.field(default=None, repr=False, init=False)

    def get_client(self):
        from dask.distributed import LocalCluster

        if self._cluster is not None:
            return self._cluster.get_client()
        if self.use_gateway:
            from dask_gateway import Gateway

            gateway = Gateway()
            options = gateway.cluster_options()
            options.update(self.cluster_options)
            self._cluster = gateway.new_cluster(options)
            self._cluster.scale(self.n_workers)
        else:
            self._cluster = LocalCluster(
                n_workers=self.n_workers, **self.cluster_options
            )
        return self._cluster.get_client()

    def close_cluster(self):
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None


@dataclasses.dataclass
class UFSReplayDatasetConfig:
    """Top-level config for UFS replay ocean preprocessing.

    Attributes:
        ocean_zarr: Full ``gs://`` path to the MOM6 zarr store.
        atmo_zarr: Full ``gs://`` path to the FV3 zarr store.
        output_directory: Where to write the output zarr.
        vertical_coarsen: Thickness-weighted vertical coarsening config.
        time_coarsen_factor: Factor by which to coarsen in time (e.g. 4 to go
            from 6-hourly to daily). 1 means no coarsening.
        ocean_rename: Additional variable renaming for the ocean store.
        atmo_forcing_vars: Mapping of FV3 var names to output forcing names.
        ice_vars: Mapping of FV3 ice var names to output names.
        stress_rename: Mapping of MOM6 stress var names to output names.
        chunking: Chunk sizes for the intermediate computation.
        sharding: Outer chunk (shard) sizes for the final zarr.
        n_split: Number of partitions for ``xpartition`` writing.
        dask: Dask cluster configuration.
    """

    ocean_zarr: str
    atmo_zarr: str
    output_directory: str

    regrid: RegridConfig = dataclasses.field(default_factory=RegridConfig)
    vertical_coarsen: VerticalCoarsenConfig = dataclasses.field(
        default_factory=VerticalCoarsenConfig
    )
    time_coarsen_factor: int = 1

    ocean_rename: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(_OCEAN_RENAME)
    )
    atmo_forcing_vars: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(_ATMO_FORCING_VARS)
    )
    ice_vars: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(_ICE_VARS)
    )
    stress_rename: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(_STRESS_RENAME)
    )

    chunking: ChunkingConfig = dataclasses.field(
        default_factory=lambda: ChunkingConfig(time_dim=1)
    )
    sharding: Optional[ChunkingConfig] = dataclasses.field(
        default_factory=lambda: ChunkingConfig(time_dim=360)
    )
    n_split: int = 10
    dask: Optional[DaskConfig] = None

    @classmethod
    def from_file(cls, path: str) -> "UFSReplayDatasetConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

# 3-D ocean variables (after renaming) that get split per level
VARS_3D = ("thetao", "so", "uo", "vo")

# MOM6 surface variables to retain (original names before renaming)
_OCEAN_SURFACE_VARS = ["SSH", "taux", "tauy"]

# MOM6 variables that are dropped (not needed for training)
_OCEAN_DROP = [
    "Heat_PmE",
    "LW",
    "LwLatSens",
    "SW",
    "depth",
    "evap",
    "fprec",
    "latent",
    "lprec",
    "lrunoff",
    "pbo",
    "sensible",
]


def _open_zarr(path: str, **kwargs) -> xr.Dataset:
    storage_options = {"token": "anon"} if path.startswith("gs://") else {}
    return xr.open_zarr(
        path,
        storage_options=storage_options,
        consolidated=True,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Horizontal regridding (mirrors scripts/era5/pipeline/xr-beam-pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N: dict[str, float] = {
    "F22.5": 22.5,
    "F90": 90,
    "F360": 360,
}


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compute cell boundaries from centers, clipping to [lo, hi]."""
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes(n: float) -> np.ndarray:
    """Compute Gaussian grid latitudes for grid number N (2N latitudes).

    Returns latitudes in degrees, sorted south-to-north.
    """
    from numpy.polynomial.legendre import leggauss

    nlat = round(2 * n)
    x, _ = leggauss(nlat)
    return np.sort(np.degrees(np.arcsin(x)))


def _make_target_grid(output_grid: str) -> xr.Dataset:
    """Create Gaussian target grid dataset for xESMF regridding."""
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
    """Build xESMF source grid descriptor from a dataset with lat/lon coords."""
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


_REGRIDDER_CACHE: dict[str, Any] = {}


def _get_regridder(
    ds: xr.Dataset,
    regrid_cfg: RegridConfig,
    source_grid: xr.Dataset | None = None,
):
    """Return a cached xESMF regridder for the given source grid and config."""
    import xesmf as xe

    if source_grid is None:
        source_grid = _make_source_grid(ds)
    src_shape = (len(source_grid["lat"]), len(source_grid["lon"]))
    cache_key = f"{regrid_cfg.output_grid}_{src_shape}"
    if cache_key not in _REGRIDDER_CACHE:
        dst = _make_target_grid(regrid_cfg.output_grid)
        _REGRIDDER_CACHE[cache_key] = xe.Regridder(
            source_grid, dst, regrid_cfg.method, periodic=True
        )
    return _REGRIDDER_CACHE[cache_key]


def _regrid_dataset(
    ds: xr.Dataset,
    regrid_cfg: RegridConfig,
    source_grid: xr.Dataset | None = None,
) -> xr.Dataset:
    """Regrid a dataset to a Gaussian grid using xESMF.

    Regrids each variable individually to keep memory usage bounded when
    the dataset contains many large 3-D fields.

    Skips regridding if ``regrid_cfg.output_grid`` is empty/None.
    """
    if not regrid_cfg.output_grid:
        return ds

    regridder = _get_regridder(ds, regrid_cfg, source_grid)

    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")

    regridded_vars: dict[str, xr.DataArray] = {}
    for name in ds.data_vars:
        regridded_vars[name] = regridder(ds[name], keep_attrs=True)

    regridded = xr.Dataset(regridded_vars, attrs=ds.attrs)
    return regridded


def _build_landsea_mask(ds_ocean: xr.Dataset, vdim: str) -> xr.Dataset:
    """Derive a 3-D land-sea mask.

    Tries ``landsea_mask`` first (available in the 1-degree zarr).
    Falls back to deriving the mask from NaNs in ``temp``/``thetao`` at t=0
    (the 0.25-degree zarr has no explicit mask variable).
    """
    if "landsea_mask" in ds_ocean:
        mask_3d = ds_ocean["landsea_mask"].astype(np.float32)
    else:
        for candidate in ("temp", "thetao", "so"):
            if candidate in ds_ocean and vdim in ds_ocean[candidate].dims:
                ref = ds_ocean[candidate].isel(time=0).compute()
                mask_3d = (~np.isnan(ref)).astype(np.float32)
                mask_3d.attrs = {"long_name": "ocean mask derived from NaN pattern"}
                print(f"  Derived 3-D mask from NaN pattern in '{candidate}'")
                break
        else:
            raise ValueError(
                "Cannot build land-sea mask: no 'landsea_mask' variable and "
                "no 3-D ocean variable with NaNs found."
            )

    mask_2d = mask_3d.isel({vdim: 0}).astype(np.float32)
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
    land_fraction = (1.0 - mask_2d).astype(np.float32)
    land_fraction.attrs = {"long_name": "land fraction", "units": "fraction"}
    sea_surface_fraction = mask_2d.copy()
    sea_surface_fraction.attrs = {
        "long_name": "sea surface fraction",
        "units": "fraction",
    }
    return xr.Dataset(
        {
            "mask_3d": mask_3d,
            "mask_2d": mask_2d,
            "land_fraction": land_fraction,
            "sea_surface_fraction": sea_surface_fraction,
        }
    )


def _ocean_weighted_mean(
    da: xr.DataArray, weights: xr.DataArray, dims: str
) -> xr.DataArray:
    """Thickness-weighted mean for ocean variables.

    Unlike ``compute_dataset.weighted_mean`` (which uses ``skipna=False``
    for the atmosphere), this version masks out NaN contributions so that
    native levels below the seafloor within a coarse group do not poison
    the average.  Where *all* levels are NaN the result is NaN.
    """
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

    Follows the same pattern as ``compute_vertical_coarsening`` in
    ``compute_dataset.py`` but uses MOM6 layer thickness (``ho``) as the
    weight instead of pressure thickness.

    For each coarse level *i* defined by ``interface_indices[i] = (start,
    end)``, the output is::

        coarsened_var_i = sum(var * ho, dim) / sum(ho, dim)

    over native levels ``[start, end)``.

    Args:
        ds: Dataset containing 3-D ocean variables and ``ho``.
        vars_3d: Names of 3-D variables to coarsen.
        interface_indices: List of ``[start, end)`` pairs defining contiguous
            groups of native levels for each coarse output level.
        vdim: Name of the vertical dimension.
        thickness_name: Name of the layer thickness variable in *ds*.

    Returns:
        Dataset with coarsened 3-D variables as per-level 2-D fields
        (``{name}_{i}``), per-level masks (``mask_{i}``), per-level depth
        scalars (``idepth_{i}``), and all 2-D variables passed through.
    """
    n_coarse = len(interface_indices)
    n_native = ds.sizes[vdim]
    depths = ds[vdim].values
    print(f"  Thickness-weighted coarsening: {n_native} → {n_coarse} levels")

    coarsened_arrays: dict[str, xr.DataArray] = {}

    for i, (start, end) in enumerate(interface_indices):
        ho_slice = ds[thickness_name].isel({vdim: slice(start, end)})

        for name in vars_3d:
            if name not in ds:
                continue
            array_slice = ds[name].isel({vdim: slice(start, end)})
            coarsened_da = _ocean_weighted_mean(array_slice, ho_slice, vdim)
            long_name = ds[name].attrs.get("long_name", name)
            coarsened_da.attrs["long_name"] = f"{long_name} level-{i}"
            coarsened_arrays[f"{name}_{i}"] = coarsened_da

        # Per-level mask: ocean where total thickness > 0
        ho_total = ho_slice.sum(vdim, skipna=False)
        mask_i = (ho_total.isel(time=0) > 0).astype(np.float32)
        mask_i.attrs = {"long_name": f"ocean mask level-{i}"}
        coarsened_arrays[f"mask_{i}"] = mask_i

        # Per-level depth: thickness-weighted mean of the native depths
        level_depths = depths[start:end]
        coarsened_arrays[f"idepth_{i}"] = xr.DataArray(
            float(np.mean(level_depths)),
            attrs={
                "units": "meters",
                "long_name": f"Depth at level-{i}",
            },
        )

    # Collect 2-D variables
    for name in ds.data_vars:
        if name in vars_3d or name == thickness_name:
            continue
        if vdim not in ds[name].dims:
            coarsened_arrays[name] = ds[name]

    # Surface mask and related fields
    mask_2d = coarsened_arrays["mask_0"].copy()
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
    coarsened_arrays["mask_2d"] = mask_2d
    land_frac = (1.0 - mask_2d).astype(np.float32)
    land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
    coarsened_arrays["land_fraction"] = land_frac
    ssf = mask_2d.copy()
    ssf.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
    coarsened_arrays["sea_surface_fraction"] = ssf

    # Drop stray coordinates (e.g. 'cftime') that can cause MergeError
    keep_coords = {"time", "lat", "lon"}
    cleaned: dict[str, xr.DataArray] = {}
    for name, da in coarsened_arrays.items():
        stray = [c for c in da.coords if c not in keep_coords and c not in da.dims]
        cleaned[name] = da.drop_vars(stray) if stray else da

    return xr.Dataset(cleaned)


def _split_3d_to_levels(
    ds: xr.Dataset,
    vars_3d: Sequence[str],
    vdim: str,
    mask_3d: xr.DataArray,
) -> xr.Dataset:
    """Split 3-D variables into per-level 2-D variables and add per-level masks."""
    n_levels = ds.sizes[vdim]
    new_vars: dict[str, xr.DataArray] = {}
    depths = ds[vdim].values

    keep_coords = {"time", "lat", "lon"}

    for var in vars_3d:
        if var not in ds:
            continue
        for i in range(n_levels):
            new_name = f"{var}_{i}"
            da = ds[var].isel({vdim: i}, drop=True)
            da = da.drop_vars([c for c in da.coords if c not in keep_coords])
            long_name = ds[var].attrs.get("long_name", var)
            da.attrs["long_name"] = f"{long_name} level-{i}"
            new_vars[new_name] = da

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

    result = xr.Dataset(new_vars)
    return result


def _average_atmo_to_ocean_cadence(
    ds_atmo: xr.Dataset,
    ocean_times: xr.DataArray,
) -> xr.Dataset:
    """Average 3-hourly atmosphere data to the 6-hourly ocean cadence.

    FV3 is 3-hourly; MOM6 is 6-hourly.  Instead of sub-sampling (which
    discards half the atmosphere snapshots), we average pairs of atmosphere
    timesteps that fall within each 6-hour ocean window.

    The ocean timestamps mark the *end* of each 6-hour window, so we use
    ``resample(time="6h", closed="right", label="right")`` to group the
    atmosphere data accordingly.  After resampling, we keep only the times
    that appear in the ocean dataset.
    """
    import pandas as pd

    atmo_start = ds_atmo.time.values[0]
    atmo_end = ds_atmo.time.values[-1]
    ocean_start = ocean_times.values[0]
    ocean_end = ocean_times.values[-1]

    # Restrict atmosphere to the temporal range covered by ocean data
    # with a small buffer for the averaging window
    ds_atmo = ds_atmo.sel(
        time=slice(
            pd.Timestamp(ocean_start) - pd.Timedelta("6h"),
            pd.Timestamp(ocean_end) + pd.Timedelta("3h"),
        )
    )

    n_atmo_before = len(ds_atmo.time)

    # Resample to 6-hourly by averaging within each window.
    # Using closed="right", label="right" so each window's timestamp aligns
    # with the ocean output time (end of the averaging window).
    ds_atmo_6h = ds_atmo.resample(time="6h", closed="right", label="right").mean()

    # Keep only times present in the ocean data
    common_times = np.intersect1d(ds_atmo_6h.time.values, ocean_times.values)
    if len(common_times) == 0:
        raise ValueError(
            "No overlapping timesteps between averaged atmosphere and ocean data. "
            f"Atmo range: {atmo_start} – {atmo_end}, "
            f"Ocean range: {ocean_start} – {ocean_end}"
        )

    ds_atmo_6h = ds_atmo_6h.sel(time=common_times)
    print(
        f"  Averaged {n_atmo_before} atmo timesteps → "
        f"{len(ds_atmo_6h.time)} 6-hourly means "
        f"({len(common_times)} / {len(ocean_times)} ocean times matched)"
    )
    return ds_atmo_6h


def compute_lazy_dataset(
    config: UFSReplayDatasetConfig,
    n_subsample: int | None = None,
) -> xr.Dataset:
    """Build the full lazy dataset ready for writing.

    Args:
        config: Preprocessing configuration.
        n_subsample: If set, only use the first *n_subsample* ocean timesteps.
            Applied early (before regridding) to keep local runs tractable.
    """
    xr.set_options(keep_attrs=True)

    # ── 1. Load ocean ─────────────────────────────────────────────────────
    print("Opening ocean zarr...")
    ds_ocean = _open_zarr(config.ocean_zarr, chunks={})
    print(f"  Ocean dims: {dict(ds_ocean.dims)}")

    if n_subsample is not None:
        ds_ocean = ds_ocean.isel(time=slice(None, n_subsample))
        print(f"  Subsampled ocean to {ds_ocean.sizes['time']} timesteps")

    # Log ocean field metadata (cell_methods) to document whether fields
    # are instantaneous snapshots or time-averaged.
    for _check_var in ("temp", "SSH", "ho"):
        if _check_var in ds_ocean:
            _cm = ds_ocean[_check_var].attrs.get("cell_methods", "not specified")
            print(f"  {_check_var} cell_methods: {_cm}")

    # The MOM6 zarr has two vertical coordinate variables that represent the
    # same axis: ``zl`` (nominal layer index) and ``z_l`` (actual depth in m).
    # We work with ``z_l`` as the primary vertical coordinate. Ensure we know
    # which dim the data variables actually use.
    vdim_candidates = [d for d in ("z_l", "zl") if d in ds_ocean.dims]
    # 3-D data vars typically use z_l
    vdim = "z_l"
    if vdim not in ds_ocean.dims:
        vdim = vdim_candidates[0]

    masks = _build_landsea_mask(ds_ocean, vdim)

    # Drop variables we don't need
    ds_ocean = ds_ocean.drop_vars(
        [v for v in _OCEAN_DROP if v in ds_ocean],
        errors="ignore",
    )
    # Drop the landsea_mask itself (we extracted what we need)
    ds_ocean = ds_ocean.drop_vars("landsea_mask", errors="ignore")
    # Drop the duplicate vertical dim (zl) if both exist.
    # ho may live on zl; reassign it to z_l before dropping zl.
    if "zl" in ds_ocean.dims and vdim != "zl":
        if "ho" in ds_ocean and "zl" in ds_ocean["ho"].dims:
            ho_on_vdim = ds_ocean["ho"].rename({"zl": vdim})
            ho_on_vdim = ho_on_vdim.assign_coords({vdim: ds_ocean[vdim].values})
            ds_ocean = ds_ocean.drop_vars("ho")
            ds_ocean["ho"] = ho_on_vdim
        ds_ocean = ds_ocean.drop_dims("zl", errors="ignore")

    # Rename ocean variables
    rename_map = {k: v for k, v in config.ocean_rename.items() if k in ds_ocean}
    ds_ocean = ds_ocean.rename(rename_map)

    # Rename stress variables
    stress_map = {k: v for k, v in config.stress_rename.items() if k in ds_ocean}
    ds_ocean = ds_ocean.rename(stress_map)

    # ── 2. Horizontal regridding ──────────────────────────────────────────
    # Separate ho before regridding to reduce peak data volume; it is
    # regridded independently and re-attached for the coarsening step.
    ho_ds = None
    if "ho" in ds_ocean and config.vertical_coarsen.vertical_coarsening_indices:
        ho_ds = ds_ocean[["ho"]]
        ds_ocean = ds_ocean.drop_vars("ho")

    if config.regrid.output_grid:
        print(f"  Regridding ocean to Gaussian grid {config.regrid.output_grid}...")
        source_grid = _make_source_grid(ds_ocean)
        ds_ocean = _regrid_dataset(ds_ocean, config.regrid, source_grid)
        print(f"  Regridded ocean to {dict(ds_ocean.sizes)}")

        if ho_ds is not None:
            print("  Regridding layer thickness (ho)...")
            ho_ds = _regrid_dataset(ho_ds, config.regrid, source_grid)
            ho_ds = ho_ds.assign_coords(
                lat=ds_ocean.lat.values, lon=ds_ocean.lon.values
            )
            print(f"  Regridded ho to {dict(ho_ds.sizes)}")

        # Rebuild masks from the regridded data's NaN pattern. Conservative
        # regridding propagates NaN for cells fully below the seafloor,
        # preserving depth-dependent bathymetry that would be lost by
        # regridding the binary mask and thresholding.
        ref_var = next(
            (
                v
                for v in ("thetao", "temp", "so")
                if v in ds_ocean and vdim in ds_ocean[v].dims
            ),
            None,
        )
        if ref_var is not None:
            ref_slice = ds_ocean[ref_var].isel(time=0).compute()
            mask_3d = (~np.isnan(ref_slice)).astype(np.float32)
            mask_3d.attrs = {"long_name": "ocean mask from regridded NaN pattern"}
            print(f"  Rebuilt 3-D mask from regridded '{ref_var}' NaN pattern")
        else:
            masks_to_regrid = masks[["mask_3d"]].astype(np.float32)
            masks_to_regrid = _regrid_dataset(
                masks_to_regrid, config.regrid, source_grid
            )
            mask_3d = (masks_to_regrid["mask_3d"] > 0.5).astype(np.float32)

        mask_2d = mask_3d.isel({vdim: 0}).astype(np.float32)
        mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
        land_frac = (1.0 - mask_2d).astype(np.float32)
        land_frac.attrs = {"long_name": "land fraction", "units": "fraction"}
        ssf = mask_2d.copy()
        ssf.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
        masks = xr.Dataset(
            {
                "mask_3d": mask_3d,
                "mask_2d": mask_2d,
                "land_fraction": land_frac,
                "sea_surface_fraction": ssf,
            }
        )

    # ── 3. Thickness-weighted vertical coarsening + split ──────────────────
    coarsen_indices = config.vertical_coarsen.vertical_coarsening_indices
    vars_3d_present = [v for v in VARS_3D if v in ds_ocean]

    if coarsen_indices:
        if ho_ds is None or "ho" not in ho_ds:
            raise ValueError(
                "Layer thickness variable 'ho' is required for thickness-"
                "weighted vertical coarsening but was not found in the ocean "
                "dataset.  Make sure 'ho' is not listed in _OCEAN_DROP."
            )
        # Merge ho back into the ocean dataset for coarsening
        ds_ocean["ho"] = ho_ds["ho"]
        indices_as_tuples = [tuple(pair) for pair in coarsen_indices]
        ds_levels = _compute_ocean_vertical_coarsening(
            ds_ocean,
            vars_3d_present,
            indices_as_tuples,
            vdim,
        )
        # masks are produced inside _compute_ocean_vertical_coarsening
        masks = ds_levels[["mask_2d", "land_fraction", "sea_surface_fraction"]]
    else:
        # No coarsening — split 3-D fields into per-level 2-D as before
        mask_3d_for_split = masks["mask_3d"]
        ds_ocean = ds_ocean.drop_vars("ho", errors="ignore")
        n_levels = ds_ocean.sizes.get(vdim, 0)
        print(f"  Working with {n_levels} vertical levels (no coarsening)")
        print("Splitting 3-D fields to per-level 2-D...")
        ds_levels = _split_3d_to_levels(
            ds_ocean, vars_3d_present, vdim, mask_3d_for_split
        )

    # Collect 2-D ocean variables (not 3-D, not ho)
    ocean_2d_names = [
        n
        for n in ds_ocean.data_vars
        if vdim not in ds_ocean[n].dims and n not in vars_3d_present and n != "ho"
    ]
    ds_ocean_2d = ds_ocean[ocean_2d_names]

    # ── 4. SST in Kelvin ──────────────────────────────────────────────────
    if "thetao_0" in ds_levels:
        sst_K = ds_levels["thetao_0"].copy() + 273.15
        sst_K.attrs = {"long_name": "Sea surface temperature", "units": "K"}
        ds_levels["sst"] = sst_K
    if "zos" in ds_ocean_2d:
        ds_ocean_2d["zos"].attrs.setdefault("long_name", "Sea Surface Height")

    # ── 6. Atmosphere forcing and sea-ice ─────────────────────────────────
    print("Opening atmosphere zarr...")
    ds_atmo = _open_zarr(config.atmo_zarr, chunks={})
    print(f"  Atmo dims: {dict(ds_atmo.dims)}")

    # Rename horizontal dims to match ocean (lat, lon)
    atmo_h_rename = {}
    if "grid_xt" in ds_atmo.dims:
        atmo_h_rename["grid_xt"] = "lon"
    if "grid_yt" in ds_atmo.dims:
        atmo_h_rename["grid_yt"] = "lat"
    if atmo_h_rename:
        ds_atmo = ds_atmo.rename(atmo_h_rename)

    # Only select the variables we need before regridding (much cheaper)
    needed_atmo_vars = (
        list(config.atmo_forcing_vars.keys())
        + list(config.ice_vars.keys())
        + (["lfrac"] if "lfrac" in ds_atmo else [])
    )
    ds_atmo = ds_atmo[[v for v in needed_atmo_vars if v in ds_atmo]]

    # Average 3-hourly atmosphere to 6-hourly ocean cadence *before*
    # regridding to keep memory bounded.
    ds_atmo = _average_atmo_to_ocean_cadence(ds_atmo, ds_ocean.time)

    # Regrid atmosphere to match ocean target grid
    if config.regrid.output_grid:
        print(f"  Regridding atmosphere to {config.regrid.output_grid}...")
        ds_atmo = _regrid_dataset(ds_atmo, config.regrid)
        # Assign exact lat/lon from the ocean grid so merge doesn't fail
        # on floating-point differences.
        ds_atmo = ds_atmo.assign_coords(
            lat=ds_ocean.lat.values, lon=ds_ocean.lon.values
        )
        print(f"  Regridded atmo to {dict(ds_atmo.sizes)}")

    # Restrict ocean data to the common time range
    common_times = ds_atmo.time.values
    ds_ocean_2d = ds_ocean_2d.sel(time=common_times)
    ds_levels = ds_levels.sel(time=common_times)

    if config.time_coarsen_factor > 1:
        ds_atmo = ds_atmo.coarsen(
            time=config.time_coarsen_factor, boundary="trim"
        ).mean()
        ds_ocean_2d = ds_ocean_2d.coarsen(
            time=config.time_coarsen_factor, boundary="trim"
        ).mean()
        ds_levels = ds_levels.coarsen(
            time=config.time_coarsen_factor, boundary="trim"
        ).mean()

    # Extract forcing variables
    forcing_source_names = [k for k in config.atmo_forcing_vars if k in ds_atmo]
    ds_forcing = ds_atmo[forcing_source_names].rename(
        {k: config.atmo_forcing_vars[k] for k in forcing_source_names}
    )

    # Extract sea-ice variables
    ice_source_names = [k for k in config.ice_vars if k in ds_atmo]
    ds_ice = ds_atmo[ice_source_names].rename(
        {k: config.ice_vars[k] for k in ice_source_names}
    )

    # Extract land fraction from FV3 if available (only when coarsening
    # did not already produce ocean-derived land_fraction)
    ds_land = xr.Dataset()
    if not coarsen_indices and "lfrac" in ds_atmo:
        ds_land["land_fraction"] = ds_atmo["lfrac"].isel(time=0, drop=True)
        ds_land["land_fraction"].attrs = {
            "long_name": "land fraction",
            "units": "fraction",
        }

    # ── 6. Merge everything ─────────────────────────────────────────────
    print("Merging datasets...")
    keep_coords = {"time", "lat", "lon"}
    to_merge = [ds_ocean_2d, ds_levels, ds_forcing, ds_ice]
    # When vertical coarsening is used, masks are already in ds_levels;
    # otherwise add them from the masks dataset.
    if not coarsen_indices:
        to_merge.append(masks[["mask_2d", "sea_surface_fraction"]])
    if len(ds_land) > 0:
        to_merge.append(ds_land)
    cleaned = []
    for d in to_merge:
        stray = [c for c in d.coords if c not in keep_coords and c not in d.dims]
        cleaned.append(d.drop_vars(stray) if stray else d)
    ds = xr.merge(cleaned, join="inner")

    # If land_fraction wasn't in coarsening output or atmo, derive from mask
    if "land_fraction" not in ds:
        ds["land_fraction"] = masks["land_fraction"]

    # ── 8. Insert NaNs on land ────────────────────────────────────────────
    mask_2d = ds["mask_2d"]
    for name in ds.data_vars:
        if name.startswith("mask_") or name.startswith("idepth_"):
            continue
        if name in ("land_fraction", "sea_surface_fraction"):
            continue
        v = ds[name]
        if "time" in v.dims and "lat" in v.dims and "lon" in v.dims:
            level_match = name.rsplit("_", 1)
            if len(level_match) == 2 and level_match[1].isdigit():
                level_idx = int(level_match[1])
                mask_name = f"mask_{level_idx}"
                if mask_name in ds:
                    ds[name] = v.where(ds[mask_name] > 0)
                    continue
            ds[name] = v.where(mask_2d > 0)

    # Fill sea ice on land with NaN, then fill NaN ice fraction with 0
    if "ocean_sea_ice_fraction" in ds:
        ds["ocean_sea_ice_fraction"] = ds["ocean_sea_ice_fraction"].where(
            mask_2d > 0, np.nan
        )
    if "HI" in ds:
        ds["HI"] = ds["HI"].where(mask_2d > 0, np.nan)
        if "ocean_sea_ice_fraction" in ds:
            ds["HI"] = ds["HI"].where(ds["ocean_sea_ice_fraction"] > 0, 0.0)

    # ── 8b. Nearest-neighbour fill for residual NaN in 2-D ocean vars ─────
    # After conservative regridding, a small number of coastal ocean cells
    # (~1% at the surface) can be NaN in 2-D variables whose native-resolution
    # land mask differs slightly from ``thetao`` (e.g. ``taux``/``tauy``).
    # Fill those with the nearest valid neighbour so the training data has
    # no unexpected NaN inside the ocean mask.
    #
    # Per-level 3-D variables (e.g. thetao_0, so_3) are NOT filled here;
    # any NaN mismatch is handled by the per-level mask.
    #
    # The NaN pattern is time-invariant, so the NN index map is computed
    # once from the first timestep and applied to all timesteps.
    from scipy.ndimage import distance_transform_edt

    # Names that are per-level 3-D splits — skip NN fill for these
    _level_var_prefixes = tuple(f"{v}_" for v in VARS_3D)

    for name in ds.data_vars:
        if name.startswith("mask_") or name.startswith("idepth_"):
            continue
        if name in ("land_fraction", "sea_surface_fraction"):
            continue
        if any(name.startswith(p) for p in _level_var_prefixes):
            continue
        v = ds[name]
        if "time" not in v.dims:
            continue

        ocean = mask_2d.values > 0
        sample = v.isel(time=0).values
        need_fill = np.isnan(sample) & ocean
        if not need_fill.any():
            continue

        n_fill = int(need_fill.sum())
        print(f"  NN-filling {n_fill} ocean cells for '{name}'")

        # Compute NN index map once from the first timestep
        valid = ~np.isnan(sample)
        _, nn_indices = distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        src_rows = nn_indices[0][need_fill]
        src_cols = nn_indices[1][need_fill]

        # Apply to all timesteps at once via numpy indexing
        data = v.values.copy()  # (time, lat, lon)
        data[:, need_fill] = data[:, src_rows, src_cols]
        ds[name] = xr.DataArray(data, dims=v.dims, coords=v.coords, attrs=v.attrs)

    # ── 9. Clean up dims / coords ─────────────────────────────────────────
    # Drop any remaining non-spatial dims that slipped through
    keep_dims = {"time", "lat", "lon"}
    drop_dims = [d for d in ds.dims if d not in keep_dims]
    if drop_dims:
        ds = ds.drop_dims(drop_dims)

    ds = ds.reset_coords(drop=True)

    # Cast to float32
    for name in ds.data_vars:
        if ds[name].dtype not in (np.float32, np.int32):
            ds[name] = ds[name].astype(np.float32)

    # ── 10. Chunking ──────────────────────────────────────────────────────
    chunks = {"time": 1, "lat": ds.sizes["lat"], "lon": ds.sizes["lon"]}
    if config.sharding is not None:
        outer = {"time": config.sharding.time_dim}
    else:
        outer = chunks
    ds = ds.chunk(outer)

    regrid_note = ""
    if config.regrid.output_grid:
        regrid_note = (
            f" Regridded to Gaussian grid {config.regrid.output_grid} "
            f"via xESMF {config.regrid.method}."
        )
    ds.attrs["history"] = (
        f"UFS GEFSv13 replay ocean dataset processed by "
        f"compute_ocean_dataset_ufs_replay.py. "
        f"Ocean source: {config.ocean_zarr}. "
        f"Atmosphere source: {config.atmo_zarr}."
        f"{regrid_note}"
    )

    print(
        f"Output dataset: {len(ds.data_vars)} variables, " f"{ds.nbytes / 1e9:.1f} GB"
    )
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--config", required=True, help="Path to YAML config file.")
@click.option("--output-store", required=True, help="Path to output zarr store.")
@click.option(
    "--debug",
    is_flag=True,
    help="Print dataset metadata instead of writing.",
)
@click.option(
    "--subsample",
    is_flag=True,
    help="Only process the first 73 timesteps.",
)
def main(config, output_store, debug, subsample):
    cfg = UFSReplayDatasetConfig.from_file(config)

    if not debug and cfg.dask is not None:
        print("Starting Dask cluster...")
        client = cfg.dask.get_client()
        print(client)
        print(client.dashboard_link)

    ds = compute_lazy_dataset(cfg, n_subsample=73 if subsample else None)

    ds = clear_compressors_encoding(ds)
    print(f"Final output size: {ds.nbytes / 1e9:.1f} GB")

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        inner_chunks = {"time": 1, "lat": ds.sizes["lat"], "lon": ds.sizes["lon"]}
        n_partitions = cfg.n_split
        ds.partition.initialize_store(output_store, inner_chunks=inner_chunks)
        for i in range(n_partitions):
            print(f"Writing partition {i + 1} / {n_partitions}")
            with ProgressBar():
                ds.partition.write(
                    output_store,
                    n_partitions,
                    ["time"],
                    i,
                    collect_variable_writes=True,
                )
    print("Done.")


if __name__ == "__main__":
    main()
