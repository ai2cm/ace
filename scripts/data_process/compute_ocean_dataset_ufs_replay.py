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
  4. Optionally coarsen the 75 vertical levels to a smaller set.
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
    """Select or coarsen from 75 MOM6 levels to a smaller set.

    Attributes:
        target_indices: Indices (0-based) of the 75 native levels to keep.
            If empty, all levels are kept.
    """

    target_indices: list[int] = dataclasses.field(default_factory=list)


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
        vertical_coarsen: Optional vertical level sub-selection.
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
    "evap",
    "fprec",
    "ho",
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


def _regrid_dataset(
    ds: xr.Dataset,
    regrid_cfg: RegridConfig,
    source_grid: xr.Dataset | None = None,
) -> xr.Dataset:
    """Regrid a dataset to a Gaussian grid using xESMF.

    Skips regridding if ``regrid_cfg.output_grid`` is empty/None.
    """
    if not regrid_cfg.output_grid:
        return ds

    import xesmf as xe

    cache_key = regrid_cfg.output_grid
    if cache_key not in _REGRIDDER_CACHE:
        if source_grid is None:
            source_grid = _make_source_grid(ds)
        dst = _make_target_grid(regrid_cfg.output_grid)
        _REGRIDDER_CACHE[cache_key] = xe.Regridder(
            source_grid, dst, regrid_cfg.method, periodic=True
        )
    regridder = _REGRIDDER_CACHE[cache_key]

    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")

    regridded = regridder(ds, keep_attrs=True)
    return regridded


def _build_landsea_mask(ds_ocean: xr.Dataset) -> xr.Dataset:
    """Derive a land-sea mask from the ``landsea_mask`` variable.

    MOM6 provides ``landsea_mask`` with dims ``(z_l, lat, lon)`` where
    1 = sea and 0 = land.  We use the surface level as the 2-D mask.
    """
    mask_3d = ds_ocean["landsea_mask"]  # (z_l, lat, lon), int32
    mask_2d = mask_3d.isel(z_l=0).astype(np.float32)
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
            "mask_3d": mask_3d.astype(np.float32),
            "mask_2d": mask_2d,
            "land_fraction": land_fraction,
            "sea_surface_fraction": sea_surface_fraction,
        }
    )


def _select_vertical_levels(
    ds: xr.Dataset,
    vdim: str,
    indices: Sequence[int],
) -> xr.Dataset:
    """Sub-select vertical levels by index."""
    if len(indices) == 0:
        return ds
    return ds.isel({vdim: list(indices)})


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

    for var in vars_3d:
        if var not in ds:
            continue
        for i in range(n_levels):
            new_name = f"{var}_{i}"
            da = ds[var].isel({vdim: i}, drop=True)
            long_name = ds[var].attrs.get("long_name", var)
            da.attrs["long_name"] = f"{long_name} level-{i}"
            new_vars[new_name] = da

    for i in range(n_levels):
        mask_i = mask_3d.isel({vdim: i}, drop=True).astype(np.float32)
        mask_i.attrs = {"long_name": f"ocean mask level-{i}"}
        new_vars[f"mask_{i}"] = mask_i

    for i, depth in enumerate(depths):
        new_vars[f"idepth_{i}"] = xr.DataArray(
            float(depth),
            attrs={"units": "meters", "long_name": f"Depth at level-{i}"},
        )

    result = xr.Dataset(new_vars)
    return result


def _subsample_atmo_to_ocean_times(
    ds_atmo: xr.Dataset,
    ocean_times: xr.DataArray,
) -> xr.Dataset:
    """Select atmosphere timesteps that align with ocean timesteps.

    FV3 is 3-hourly; MOM6 is 6-hourly. We select only exact matches to
    avoid duplicate-index issues from ``method='nearest'``.
    """
    common_times = np.intersect1d(ds_atmo.time.values, ocean_times.values)
    if len(common_times) == 0:
        raise ValueError(
            "No overlapping timesteps between atmosphere and ocean data. "
            f"Atmo range: {ds_atmo.time.values[0]} – {ds_atmo.time.values[-1]}, "
            f"Ocean range: {ocean_times.values[0]} – {ocean_times.values[-1]}"
        )
    print(
        f"  {len(common_times)} / {len(ocean_times)} ocean times "
        f"have exact matches in atmo data"
    )
    return ds_atmo.sel(time=common_times)


def compute_lazy_dataset(config: UFSReplayDatasetConfig) -> xr.Dataset:
    """Build the full lazy dataset ready for writing."""
    xr.set_options(keep_attrs=True)

    # ── 1. Load ocean ─────────────────────────────────────────────────────
    print("Opening ocean zarr...")
    ds_ocean = _open_zarr(config.ocean_zarr, chunks={})
    print(f"  Ocean dims: {dict(ds_ocean.dims)}")

    # The MOM6 zarr has two vertical coordinate variables that represent the
    # same axis: ``zl`` (nominal layer index) and ``z_l`` (actual depth in m).
    # We work with ``z_l`` as the primary vertical coordinate. Ensure we know
    # which dim the data variables actually use.
    vdim_candidates = [d for d in ("z_l", "zl") if d in ds_ocean.dims]
    # 3-D data vars typically use z_l
    vdim = "z_l"
    if vdim not in ds_ocean.dims:
        vdim = vdim_candidates[0]

    masks = _build_landsea_mask(ds_ocean)

    # Drop variables we don't need
    ds_ocean = ds_ocean.drop_vars(
        [v for v in _OCEAN_DROP if v in ds_ocean],
        errors="ignore",
    )
    # Drop the landsea_mask itself (we extracted what we need)
    ds_ocean = ds_ocean.drop_vars("landsea_mask", errors="ignore")
    # Drop the duplicate vertical dim (zl) if both exist
    if "zl" in ds_ocean.dims and vdim != "zl":
        ds_ocean = ds_ocean.drop_dims("zl", errors="ignore")

    # Rename ocean variables
    rename_map = {k: v for k, v in config.ocean_rename.items() if k in ds_ocean}
    ds_ocean = ds_ocean.rename(rename_map)

    # Rename stress variables
    stress_map = {k: v for k, v in config.stress_rename.items() if k in ds_ocean}
    ds_ocean = ds_ocean.rename(stress_map)

    # ── 2. Horizontal regridding ──────────────────────────────────────────
    if config.regrid.output_grid:
        print(f"  Regridding ocean to Gaussian grid {config.regrid.output_grid}...")
        source_grid = _make_source_grid(ds_ocean)
        ds_ocean = _regrid_dataset(ds_ocean, config.regrid, source_grid)
        # Regrid 3-D mask separately (it has no time dim)
        masks_to_regrid = masks[["mask_3d"]].astype(np.float32)
        masks_to_regrid = _regrid_dataset(masks_to_regrid, config.regrid, source_grid)
        # Threshold regridded mask back to binary
        masks["mask_3d"] = (masks_to_regrid["mask_3d"] > 0.5).astype(np.float32)
        # Rebuild 2-D masks from regridded 3-D mask
        mask_2d = masks["mask_3d"].isel({vdim: 0}).astype(np.float32)
        mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
        masks["mask_2d"] = mask_2d
        masks["land_fraction"] = (1.0 - mask_2d).astype(np.float32)
        masks["land_fraction"].attrs = {
            "long_name": "land fraction",
            "units": "fraction",
        }
        masks["sea_surface_fraction"] = mask_2d.copy()
        masks["sea_surface_fraction"].attrs = {
            "long_name": "sea surface fraction",
            "units": "fraction",
        }
        print(f"  Regridded ocean to {dict(ds_ocean.sizes)}")

    # ── 3. Vertical sub-selection ──────────────────────────────────────────
    target_indices = config.vertical_coarsen.target_indices
    if target_indices:
        print(f"  Selecting {len(target_indices)} of {ds_ocean.sizes[vdim]} levels")
        ds_ocean = _select_vertical_levels(ds_ocean, vdim, target_indices)
        masks["mask_3d"] = _select_vertical_levels(
            masks[["mask_3d"]], vdim, target_indices
        )["mask_3d"]
    n_levels = ds_ocean.sizes.get(vdim, 0)
    print(f"  Working with {n_levels} vertical levels")

    # ── 4. Split 3-D → per-level 2-D ─────────────────────────────────────
    print("Splitting 3-D fields to per-level 2-D...")
    vars_3d_present = [v for v in VARS_3D if v in ds_ocean]
    ds_levels = _split_3d_to_levels(ds_ocean, vars_3d_present, vdim, masks["mask_3d"])

    # Collect 2-D ocean variables
    ocean_2d_names = [
        n
        for n in ds_ocean.data_vars
        if vdim not in ds_ocean[n].dims and n not in vars_3d_present
    ]
    ds_ocean_2d = ds_ocean[ocean_2d_names]

    # ── 5. SST in Kelvin ─────────────────────────────────────────────────
    # ``thetao`` after rename is potential temperature in degC
    if "thetao_0" in ds_levels:
        sst_K = ds_levels["thetao_0"].copy() + 273.15
        sst_K.attrs = {"long_name": "Sea surface temperature", "units": "K"}
        ds_levels["sst"] = sst_K
    # Also rename ``zos`` from ocean_2d if present
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

    # Regrid atmosphere to match ocean target grid
    if config.regrid.output_grid:
        # Only select the variables we need before regridding (much cheaper)
        needed_atmo_vars = (
            list(config.atmo_forcing_vars.keys())
            + list(config.ice_vars.keys())
            + (["lfrac"] if "lfrac" in ds_atmo else [])
        )
        ds_atmo = ds_atmo[[v for v in needed_atmo_vars if v in ds_atmo]]
        print(f"  Regridding atmosphere to {config.regrid.output_grid}...")
        ds_atmo = _regrid_dataset(ds_atmo, config.regrid)
        print(f"  Regridded atmo to {dict(ds_atmo.sizes)}")

    # Sub-sample atmo to ocean times
    ds_atmo = _subsample_atmo_to_ocean_times(ds_atmo, ds_ocean.time)

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

    # Extract land fraction from FV3 if available
    ds_land = xr.Dataset()
    if "lfrac" in ds_atmo:
        ds_land["land_fraction"] = ds_atmo["lfrac"].isel(time=0, drop=True)
        ds_land["land_fraction"].attrs = {
            "long_name": "land fraction",
            "units": "fraction",
        }

    # ── 7. Merge everything ───────────────────────────────────────────────
    print("Merging datasets...")
    ds = xr.merge(
        [
            ds_ocean_2d,
            ds_levels,
            ds_forcing,
            ds_ice,
            masks[["mask_2d", "sea_surface_fraction"]],
            ds_land if len(ds_land) > 0 else xr.Dataset(),
        ],
        join="inner",
    )

    # If land_fraction wasn't in atmo, derive from mask
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

    ds = compute_lazy_dataset(cfg)

    if subsample:
        ds = ds.isel(time=slice(None, 73))

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
