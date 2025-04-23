"""
Utils for CM4 ocean data preprocessing. This script relies on a fork of the
m2lines/ocean_emulators repo:
https://github.com/jpdunc23/ocean_emulators/tree/cm4-preprocessing.

The 200-year pre-industrial control simulation ocean preprocessing ran in about
6 hours on LEAP's 2i2c JupyterHub in the default "notebook" conda environment
using Dask Gateway with 64x '8CPU, 57.9Gi' workers. See the
CM4-pre-industrial-control-ocean-200yr.yaml and this notebook for details:
https://github.com/ai2cm/explore/blob/master/elynn/2024-11-11-CM4-200yr-preprocessing-ocean.ipynb
"""

import dataclasses
import os
import pdb
import sys
from typing import IO, Any, Mapping, Optional, Protocol, Sequence, Tuple, Union

import click
import dacite
import fsspec
import numpy as np
import xarray as xr
import xpartition  # noqa
import yaml
from compute_dataset import ChunkingConfig, StandardDimMapping
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from get_stats import StatsConfig
from ocean_emulators.preprocessing import (
    horizontal_regrid,
    rotate_vectors,
    spatially_filter,
)
from ocean_emulators.simulation_preprocessing.gfdl_cm4 import cm4_preprocessing


@dataclasses.dataclass
class OceanStandardNameMapping(StandardDimMapping):
    longitude_dim: str = "x"
    latitude_dim: str = "y"
    time_dim: str = "time"
    vertical_dim: str = "lev"
    vertical_idim: str = "ilev"
    rotation_angle: str = "angle"
    sea_water_x_velocity: str = "uo"
    sea_water_y_velocity: str = "vo"
    sea_water_salinity: str = "so"
    sea_water_potential_temperature: str = "thetao"
    surface_temperature: str = "tos"
    surface_downward_x_stress: str = "tauuo"
    surface_downward_y_stress: str = "tauvo"
    sea_ice_x_velocity: str = "UI"
    sea_ice_y_velocity: str = "VI"
    sea_ice_modeled: str = "EXT"
    sea_ice_fraction: str = "sea_ice_fraction"
    wetmask: str = "wetmask"
    ocean_layer_thickness: str = "layer_thickness"

    def __post_init__(self):
        self.full_field_dims = [self.longitude_dim, self.latitude_dim, self.time_dim]

    @property
    def rotated_vars(self) -> Sequence[Tuple[str, str]]:
        return (
            (self.sea_water_x_velocity, self.sea_water_y_velocity),
            (self.sea_ice_x_velocity, self.sea_ice_y_velocity),
            (self.surface_downward_x_stress, self.surface_downward_y_stress),
        )

    @property
    def vars_3d(self) -> Sequence[str]:
        return (
            self.sea_water_x_velocity,
            self.sea_water_y_velocity,
            self.sea_water_salinity,
            self.sea_water_potential_temperature,
        )


def _rename(ds: xr.Dataset, renaming: Mapping[str, str]):
    if len(renaming) == 0:
        return ds
    return ds.rename(renaming)


@dataclasses.dataclass
class CoarseningConfig:
    """Configuration for coarsening a supplemental dataset with higher temporal
    frequency. Assumed to have the same horizontal coordinates as the other
    datasets.

    Attributes:
        zarr: name of zarr with the higher frequency data.
        n_coarsen: number of timepoints over which to take the mean.
        renaming: (optional) mapping of names in dataset to renamed output.

    """

    zarr: str
    n_coarsen: int
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def coarsen(self, ds: xr.Dataset, time_dim: str):
        return ds.coarsen({time_dim: self.n_coarsen}).mean().drop_vars(time_dim)

    def rename(self, ds: xr.Dataset):
        return _rename(ds, self.renaming)


@dataclasses.dataclass
class OceanStaticConfig:
    """Configuration for static ocean data.

    Attributes:
        zarr: name of zarr with the static ocean data.
        names: names of variables to extract from the zarr.
        renaming: (optional) mapping of names in dataset to renamed output.


    """

    zarr: str
    names: Sequence[str]
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def rename(self, ds: xr.Dataset):
        return _rename(ds, self.renaming)


@dataclasses.dataclass
class DaskConfig:
    """Configuration for Dask, either LocalCluster or dask-gateway. See
    https://docs.2i2c.org/user/howto/launch-dask-gateway-cluster for
    dask-gateway usage on the LEAP-Pangeo 2i2c hub, where the relevant option is
    "worker_resource_allocation" which can be one of "1CPU, 7.2Gi", "2CPU, 14.5Gi",
    "4CPU, 28.9Gi", "8CPU, 57.9Gi", and "16CPU, 115.8Gi". Prepend
    https://leap.2i2c.cloud to dask dashboard urls when using the LEAP-Pangeo
    2i2c hub.

    Attributes:
        n_workers: number of Dask workers.
        use_gateway: whether to use dask-gateway
        cluster_options: additional options for configuring the LocalCluster or
            Gateway.

    """

    n_workers: int = 16
    use_gateway: bool = False
    cluster_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    _cluster = None

    def get_client(self) -> Client:
        if self._cluster is not None:
            return self._cluster.get_client()
        if self.use_gateway:
            from dask_gateway import Gateway

            # use default gateway settings
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
class OceanDatasetComputationConfig:
    """Configuration of computation details for an FME reference dataset.

    Attributes:
        ocean_zarr: name of zarr with ocean model outputs.
        ice_zarr: name of zarr with ice model outputs.
        nc_grid_path: path to nc file with variables needed by
            ocean_emulators.simulation_processing.gfdl_om4.om4_preprocessing.
        nc_mosaic_path: path to nc file for grid mosaic with variables needed by
            ocean_emulators.simulation_processing.gfdl_om4.convert_super_grid.
        nc_target_grid_path: path to nc file for target grid.
        ocean_static_zarr: (optional) name of zarr with static ocean data.
        renaming: (optional) mapping of names in dataset to renamed output.
        standard_names: (optional) mapping of standard names to corresponding
            names of variables in the dataset.
        chunking: (optional) config for dask chunking.
    """

    ocean_zarr: str
    ice_zarr: str
    nc_grid_path: str
    nc_mosaic_path: str
    nc_target_grid_path: str
    coarsen: Optional[CoarseningConfig] = None
    ocean_static: Optional[OceanStaticConfig] = None
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)
    standard_names: OceanStandardNameMapping = dataclasses.field(
        default_factory=OceanStandardNameMapping
    )
    chunking: ChunkingConfig = dataclasses.field(default_factory=ChunkingConfig)
    ocean_dataset_nc_files: str = ""
    ocean_dataset_monthly_layer_thickness_files: str = ""
    compute_e3sm_surface_downward_heat_flux: bool = False
    ice_dataset_nc_files: str = ""
    ocean_vertical_target_interface_levels: list = dataclasses.field(
        default_factory=list
    )

    def rename(self, ds: xr.Dataset):
        return _rename(ds, self.renaming)


class FileSystemProtocol(Protocol):
    def get_mapper(self, url: str) -> Union[str, fsspec.mapping.FSMap]: ...

    def open(self, *args, **kwargs) -> IO: ...


class _DummyFileSystem(FileSystemProtocol):
    def get_mapper(self, url: str) -> str:
        return url

    def open(self, *args, **kwargs) -> IO:
        return open(*args, **kwargs)


_DUMMY_FS = _DummyFileSystem()


@dataclasses.dataclass
class FileSystemConfig(FileSystemProtocol):
    """Configuration for creating an fsspec.filesystem.

    Attributes:
        protocol: file system protocol.
        fs_options: options for file system.

    """

    protocol: str
    storage_options: Mapping[str, Any]

    def __post_init__(self):
        self._fs = fsspec.filesystem(self.protocol, **self.storage_options)

    def get_mapper(self, url: str) -> fsspec.mapping.FSMap:
        return self._fs.get_mapper(url)


@dataclasses.dataclass
class OceanDatasetConfig:
    """Dataset provenance for a set of reference simulations.

    Attributes:
        runs: mapping of short names to full paths of reference datasets.
        data_output_directory: path to parent directory where the output zarr
            will be written.
        dataset_computation: configuration details for compute_lazy_dataset.
        stats_config: configuration to retrieve statistics dataset.
        n_split: number of xpartition partitions to use when writing the data.
        dask: (optional) configuration for the dask cluster.
        filesystem: (optional) file system to use for reading and writing data.

    """

    runs: Mapping[str, str]
    data_output_directory: str
    dataset_computation: OceanDatasetComputationConfig
    stats: StatsConfig
    n_split: int = 10
    dask: Optional[DaskConfig] = None
    filesystem: Optional[FileSystemConfig] = None

    @classmethod
    def from_file(cls, path: str) -> "OceanDatasetConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


def compute_lazy_dataset(
    config: OceanDatasetComputationConfig,
    run_directory: str,
    fs: Optional[FileSystemProtocol] = None,
    backend_kwargs=None,
):
    """Compute CM4 ocean and ice dataset.

    Arguments:
        config: Path to dataset configuration YAML file.
        run_directory: Path to reference run directory.
    """
    if fs is None:
        fs = _DUMMY_FS

    xr.set_options(keep_attrs=True)

    ocean_names = config.standard_names
    time_dim = ocean_names.time_dim
    lat_dim = ocean_names.latitude_dim
    lon_dim = ocean_names.longitude_dim
    vdim = ocean_names.vertical_dim
    vidim = ocean_names.vertical_idim

    om_zarr_path = os.path.join(run_directory, config.ocean_zarr)
    sis_zarr_path = os.path.join(run_directory, config.ice_zarr)

    ds = cm4_preprocessing(
        om_zarr_path=om_zarr_path,
        sis_zarr_path=sis_zarr_path,
        nc_grid_path=config.nc_grid_path,
        nc_mosaic_path=config.nc_mosaic_path,
        fs=fs,
        backend_kwargs=backend_kwargs,
    )
    ds = ds.chunk(
        {
            vdim: 1,
            lat_dim: -1,
            lon_dim: -1,
        }
    )

    urls = [
        om_zarr_path,
        sis_zarr_path,
        config.nc_grid_path,
        config.nc_mosaic_path,
        config.nc_target_grid_path,
    ]

    if config.coarsen is not None:
        zarr_path = fs.get_mapper(os.path.join(run_directory, config.coarsen.zarr))
        urls.append(zarr_path)
        ds_coarsen = xr.open_zarr(zarr_path)
        ds_coarsen = config.coarsen.coarsen(ds_coarsen, time_dim)
        ds_coarsen = config.coarsen.rename(ds_coarsen)
        ds_coarsen = ds_coarsen.chunk(
            {time_dim: ds.chunks[time_dim], lon_dim: -1, lat_dim: -1}
        )
        ds = xr.merge([ds, ds_coarsen])

    if config.ocean_static is not None:
        zarr_path = os.path.join(run_directory, config.ocean_static.zarr)
        urls.append(zarr_path)
        with fs.open(zarr_path) as f:
            ds_static = xr.open_dataset(
                zarr_path,
                decode_timedelta=False,
                engine="zarr",
                backend_kwargs=backend_kwargs,
            )[config.ocean_static.names]
        ds_static = config.ocean_static.rename(ds_static)
        ds = xr.merge([ds, ds_static])

    idepth_data = {}

    for i, depth in enumerate(ds[vidim].values):
        idepth_data[f"idepth_{i}"] = xr.DataArray(depth)
        idepth_data[f"idepth_{i}"].attrs["units"] = "meters"
        idepth_data[f"idepth_{i}"].attrs["long_name"] = f"Depth at interface level-{i}"

    idepth_ds = xr.Dataset(idepth_data)

    assert (
        ocean_names.sea_ice_fraction in ds
    ), f"Sea ice fraction variable {ocean_names.sea_ice_fraction} is missing."

    print(f"Preprocessed size: {ds.nbytes / 1e9:.1f} GB")

    # save attributes to add back after processing
    attrs = {}
    for var in ds.data_vars:
        attrs[var] = ds[var].attrs

    angle = ds[ocean_names.rotation_angle]
    for varname_x, varname_y in ocean_names.rotated_vars:
        x_rotated, y_rotated = rotate_vectors(ds[varname_x], ds[varname_y], angle)
        ds[varname_x] = x_rotated.astype(np.float32)
        ds[varname_y] = y_rotated.astype(np.float32)

    # spatial filtering
    ds = spatially_filter(
        ds, ds[ocean_names.wetmask], depth_dim=vdim, y_dim=lat_dim, x_dim=lon_dim
    )

    # regrid
    with fs.open(config.nc_target_grid_path) as f:
        ds_target_grid = xr.open_dataset(f).load()

    # TODO: remove target grid dimension assumptions
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
    # fill nans in sea_ice_fraction to be
    # consistent with ocean fraction in ocean_emulators
    if ocean_names.sea_ice_fraction in ds.data_vars:
        ds[ocean_names.sea_ice_fraction] = ds[ocean_names.sea_ice_fraction].fillna(0.0)
    ds_regridded = horizontal_regrid(ds, ds_target_grid).astype("float32")
    if ocean_names.sea_ice_fraction in ds_regridded.data_vars:
        ds_regridded[ocean_names.sea_ice_fraction] = ds_regridded[
            ocean_names.sea_ice_fraction
        ].fillna(0.0)
    print(f"Regridded size: {ds_regridded.nbytes / 1e9:.1f} GB")

    ds = ds_regridded
    for var, attrs in attrs.items():
        ds[var].attrs = attrs

    # fill ice velocity with NaN where sea ice is 0
    cond = ds[ocean_names.sea_ice_modeled] > 0.0

    for var in [ocean_names.sea_ice_x_velocity, ocean_names.sea_ice_y_velocity]:
        ds[var] = ds[var].where(cond, np.nan)

    wetmask = ds[ocean_names.wetmask].astype(np.float32)
    if len(wetmask.attrs) == 0:
        wetmask.attrs["long_name"] = "ocean mask"
        wetmask.attrs["units"] = "0 if land, 1 if ocean"

    ds["mask"] = wetmask
    vars_3d = list(ocean_names.vars_3d) + ["mask"]

    for i, _ in enumerate(ds[vdim].values):
        for var in vars_3d:
            long_name = ds[var].long_name
            ds[f"{var}_{i}"] = ds[var].isel({vdim: i})
            ds[f"{var}_{i}"].attrs["long_name"] = long_name + f" level-{i}"

    ds = ds.drop_vars(vars_3d)
    ds = ds.drop_dims(vdim)
    ds = ds.reset_coords(drop=True)
    ds = xr.merge([ds, idepth_ds])

    # add 'sst' variable in degrees Kelvin
    ds["sst"] = ds[ocean_names.surface_temperature].copy() + 273.15
    ds["sst"].attrs["long_name"] = "Sea surface temperature"
    ds["sst"].attrs["units"] = "K"

    chunks = config.chunking.get_chunks(ocean_names)
    ds = ds.chunk(chunks)
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts/data_process"
        "/compute_ocean_dataset.py"
        f" script, using following input sources: {urls}."
    )

    drop_dims = [x for x in list(ds.dims) if x not in ocean_names.full_field_dims]
    ds = ds.drop_dims(drop_dims)
    ds = config.rename(ds)

    return ds


@click.command()
@click.option("--config", help="Path to dataset configuration YAML file.")
@click.option("--run-directory", help="Path to reference run directory.")
@click.option("--output-store", help="Path to output zarr store.")
@click.option(
    "--debug",
    is_flag=True,
    help="Print metadata and return QC plots instead of writing output.",
)
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
def main(
    config,
    run_directory,
    output_store,
    debug,
    subsample,
):
    print(f"--run-directory is {run_directory}")
    print(f"--output-store is {output_store}")

    config = OceanDatasetConfig.from_file(config)

    if not debug and config.dask is not None:
        print("Using dask cluster...")
        client = config.dask.get_client()
        print(client)
        print(client.dashboard_link)

    ds = compute_lazy_dataset(
        config=config.dataset_computation,
        run_directory=run_directory,
        fs=config.filesystem,
    )

    if subsample:
        ds = ds.isel(time=slice(None, 73))

    print(f"Output dataset size is {ds.nbytes / 1e9} GB")

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        n_partitions = config.n_split
        ds.partition.initialize_store(output_store)
        for i in range(n_partitions):
            print(f"Writing segment {i + 1} / {n_partitions}")
            with ProgressBar():
                ds.partition.write(
                    output_store,
                    n_partitions,
                    ["time"],
                    i,
                    collect_variable_writes=True,
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
