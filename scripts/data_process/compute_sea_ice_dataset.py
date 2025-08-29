"""
Utils for 6-hourly CM4 sea ice data preprocessing. This script relies on
the ai2cm/ocean_emulators fork of m2lines/ocean_emulators.

"""

import dataclasses
import os
from typing import Mapping, Sequence

import dacite
import numpy as np
import xarray as xr
import yaml
from compute_dataset import ChunkingConfig, StandardDimMapping
from compute_ocean_dataset import DaskConfig, FileSystemConfig, StaticDataConfig
from ocean_emulators.preprocessing import horizontal_regrid, rotate_vectors
from ocean_emulators.simulation_preprocessing.gfdl_cm4 import sis2_preprocessing
from ocean_emulators.simulation_preprocessing.gfdl_om4 import convert_super_grid


@dataclasses.dataclass
class SeaIceStandardNameMapping(StandardDimMapping):
    longitude_dim: str = "x"
    latitude_dim: str = "y"
    time_dim: str = "time"
    air_stress_on_ice_x_component: str = "FA_X"
    air_stress_on_ice_y_component: str = "FA_Y"
    sea_ice_fraction: str = "sea_ice_fraction"
    sea_surface_fraction: str = "sea_surface_fraction"
    sea_surface_mask: str = "wet"

    def __post_init__(self):
        self.full_field_dims = [self.longitude_dim, self.latitude_dim, self.time_dim]

    @property
    def rotated_vars(self) -> Sequence[tuple[str, str]]:
        return (
            (self.air_stress_on_ice_x_component, self.air_stress_on_ice_y_component),
        )


@dataclasses.dataclass
class SeaIceDatasetComputationConfig:
    """Configuration of computation details for an FME reference dataset.

    Attributes:
        ice_zarr: name of zarr with ice model outputs.
        nc_target_grid_path: path to nc file for target grid.
        ocean_static: config for zarr with static ocean data, include surface mask.
        outer_chunks: (optional) config for dask chunking.
        sharding: (optional) mapping of standard dimension names to desired
            output shard sizes. Defaults to a shard size of 360 along the time
            dimension. If None, then an unsharded zarr store will be written
            with chunks as specified in ``chunking``.
        shift_timestamps_to_avg_interval_midpoint: If True, shift time axis labels
            backwards by half the ocean step size.
    """

    ice_zarr: str
    nc_mosaic_path: str
    nc_target_grid_path: str
    ocean_static: StaticDataConfig
    outer_chunks: ChunkingConfig = dataclasses.field(
        default_factory=lambda: ChunkingConfig(time_dim=360)
    )


def compute_lazy_dataset(
    config: SeaIceDatasetComputationConfig,
    run_directory: str,
    fs=None,
    backend_kwargs=None,
):
    """Compute CM4 sea ice dataset.

    Arguments:
        config: Configuration for sea ice dataset computation.
        run_directory: Path to reference run directory.
        fs: Optional file system interface.
        backend_kwargs: Optional backend kwargs for zarr loading.
    """
    if fs is None:
        from compute_ocean_dataset import _DUMMY_FS

        fs = _DUMMY_FS

    xr.set_options(keep_attrs=True)

    sea_ice_names = SeaIceStandardNameMapping()
    lat_dim = sea_ice_names.latitude_dim
    lon_dim = sea_ice_names.longitude_dim

    # Step 1: Preprocess ice_zarr using sis2_preprocessing
    ice_zarr_path = os.path.join(run_directory, config.ice_zarr)
    ds = sis2_preprocessing(zarr_data_path=ice_zarr_path, backend_kwargs=backend_kwargs)
    ds = ds.chunk(
        {
            lat_dim: -1,
            lon_dim: -1,
        }
    )

    # Load mosaic dataset
    with fs.open(config.nc_mosaic_path) as f:
        ds_mosaic = xr.open_dataset(f).load()

    ds_mosaic = xr.Dataset(
        dict(
            zip(
                ["angle", "lon", "lat", "lon_b", "lat_b"],
                convert_super_grid(ds_mosaic),
            )
        )
    ).rename({"xh": "x", "yh": "y", "xh_b": "x_b", "yh_b": "y_b"})
    mosaic_dict = dict(ds_mosaic)
    angle = mosaic_dict.pop("angle")
    ds = ds.assign_coords(mosaic_dict)

    # save attributes to add back after processing
    attrs = {}
    for var in ds.data_vars:
        attrs[var] = ds[var].attrs

    for varname_x, varname_y in sea_ice_names.rotated_vars:
        if varname_x in ds.data_vars and varname_y in ds.data_vars:
            x_rotated, y_rotated = rotate_vectors(ds[varname_x], ds[varname_y], angle)
            ds[varname_x] = x_rotated
            ds[varname_y] = y_rotated

    ocean_static_zarr_path = os.path.join(run_directory, config.ocean_static.zarr)
    with fs.open(ocean_static_zarr_path) as f:
        ds_ocean_static = xr.open_dataset(
            ocean_static_zarr_path,
            decode_timedelta=False,
            engine="zarr",
            backend_kwargs=backend_kwargs,
        )[config.ocean_static.names]
    ds_ocean_static = config.ocean_static.rename(ds_ocean_static)

    # mask => sea_surface_fraction (after regridding)
    ds[sea_ice_names.sea_surface_fraction] = ds_ocean_static[
        sea_ice_names.sea_surface_mask
    ]

    # Load target grid dataset
    with fs.open(config.nc_target_grid_path) as f:
        ds_target_grid = xr.open_dataset(f).load()

    ds_target_grid = ds_target_grid.rename(
        {
            "grid_x": "x_b",
            "grid_y": "y_b",
            "grid_xt": lon_dim,
            "grid_yt": lat_dim,
            "grid_lon": "lon_b",
            "grid_lat": "lat_b",
            "grid_lont": "lon",
            "grid_latt": "lat",
        }
    )

    ds[sea_ice_names.sea_ice_fraction] = ds[sea_ice_names.sea_ice_fraction].fillna(0.0)
    ds = horizontal_regrid(
        ds, ds_target_grid, wetmask_name=sea_ice_names.sea_surface_fraction
    )
    for var, attrs in attrs.items():
        ds[var].attrs = attrs

    ds[sea_ice_names.sea_surface_fraction] = ds[
        sea_ice_names.sea_surface_fraction
    ].fillna(0.0)

    insert_nan_names = [
        x for x in ds.data_vars if x != sea_ice_names.sea_surface_fraction
    ]
    for name in insert_nan_names:
        ds[name] = ds[name].where(ds[sea_ice_names.sea_surface_fraction] > 0)

    urls = [
        ice_zarr_path,
        ocean_static_zarr_path,
        config.nc_mosaic_path,
        config.nc_target_grid_path,
    ]
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts/data_process"
        "/compute_sea_ice_dataset.py"
        f" script, using following input sources: {urls}."
    )
    drop_coords = [x for x in list(ds.coords) if x not in sea_ice_names.full_field_dims]
    ds = ds.reset_coords().drop(drop_coords)

    outer_chunks = config.outer_chunks.get_chunks(sea_ice_names)
    ds = ds.chunk(outer_chunks)

    ds = ds.rename({lon_dim: "lon", lat_dim: "lat"})
    return ds.astype(np.float32)


@dataclasses.dataclass
class SeaIceDatasetConfig:
    """Dataset provenance for a set of reference simulations.

    Attributes:
        runs: mapping of short names to full paths of reference datasets.
        dataset_computation: configuration details for compute_lazy_dataset.
        n_split: number of xpartition partitions to use when writing the data.
        dask: (optional) configuration for the dask cluster.
        filesystem: (optional) file system to use for reading and writing data.

    """

    runs: Mapping[str, str]
    dataset_computation: SeaIceDatasetComputationConfig
    n_split: int = 10
    dask: DaskConfig | None = None
    filesystem: FileSystemConfig | None = None

    @classmethod
    def from_file(cls, path: str) -> "SeaIceDatasetConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )
