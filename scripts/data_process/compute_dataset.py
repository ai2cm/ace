# This script is used to compute a training dataset from the "raw"
# FV3GFS data stored in zarr form on GCS.

# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

# The resulting dataset is about 194GB (the input is about 2.5TB). Running this script
# on my 8-CPU VM takes about 2.5 hours. See "compute_dataset_fv3gfs_argo_workflow.yaml"
# for a workflow which parallelizes this script across the 11-member ensemble and runs
# it on our GKE cluster.

import abc
import dataclasses
import os
import sys
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import click
import dacite
import fsspec
import numpy as np
import xarray as xr
import xpartition  # noqa: F401
import xtorch_harmonics
import yaml
from dask.diagnostics import ProgressBar

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_stats import StatsConfig

# constants are defined as in FV3GFS model
# https://github.com/ai2cm/fv3gfs-fortran/blob/master/FMS/constants/constants.F90
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg
GRAVITY = 9.80665  # m/s^2

SPECIFIC_TOTAL_WATER = "specific_total_water"
# following name is for the water path computed by vertically integrating the
# lat-lon 3D variables. This is the water path we want to use for training. It is not
# necessarily exactly equal to the precipitable water path computed online by FV3GFS.
TOTAL_WATER_PATH = "total_water_path"


@dataclasses.dataclass
class StandardNameMapping:
    longitude_dim: str = "grid_xt"
    latitude_dim: str = "grid_yt"
    vertical_dim: str = "pfull"
    vertical_interface_dim: str = "phalf"
    time_dim: str = "time"
    surface_pressure: str = "PRESsfc"
    latent_heat_flux: str = "LHTFLsfc"
    precip_rate: str = "PRATEsfc"
    precipitable_water_path: str = "precipitable_water_path"
    pressure_thickness: str = "pressure_thickness_of_atmospheric_layer"
    air_temperature: str = "air_temperature"
    specific_humidity: str = "specific_humidity"
    cloud_water_mixing_ratio: str = "cloud_water_mixing_ratio"
    cloud_ice_mixing_ratio: str = "cloud_ice_mixing_ratio"
    graupel_mixing_ratio: str = "graupel_mixing_ratio"
    rain_mixing_ratio: str = "rain_mixing_ratio"
    snow_mixing_ratio: str = "snow_mixing_ratio"
    northward_wind: str = "northward_wind"
    eastward_wind: str = "eastward_wind"
    surface_evaporation_rate: str = "surface_evaporation_rate"
    land_fraction: str = "land_fraction"
    ocean_fraction: str = "ocean_fraction"
    sea_ice_fraction: str = "sea_ice_fraction"
    hybrid_level_coeffs: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.horizontal_dims: List[str] = [self.longitude_dim, self.latitude_dim]

        self.specific_total_water = SPECIFIC_TOTAL_WATER
        self.total_water_path = TOTAL_WATER_PATH
        self.pwat_tendency = f"tendency_of_{self.total_water_path}"
        self.time_derivative_names = [self.total_water_path]

        self.vertically_resolved: List[str] = [
            self.specific_total_water,
            self.air_temperature,
            self.northward_wind,
            self.eastward_wind,
        ]

        # variables to drop after all derived variables are computed
        self.dropped_variables: List[str] = (
            self.water_species
            + self.vertically_resolved
            + [self.pressure_thickness, self.vertical_dim]
        )
        if self.precipitable_water_path.lower() != "none":
            self.dropped_variables.append(self.precipitable_water_path)

    @property
    def water_species(self) -> List[str]:
        return [
            item
            for item in [
                self.specific_humidity,
                self.cloud_water_mixing_ratio,
                self.cloud_ice_mixing_ratio,
                self.graupel_mixing_ratio,
                self.rain_mixing_ratio,
                self.snow_mixing_ratio,
            ]
            if item.lower() != "none"
        ]


@dataclasses.dataclass
class DLWPNameMapping(StandardNameMapping):
    longitude_dim: str = "longitude"
    latitude_dim: str = "latitude"
    face_dim: str = "face"
    width_dim: str = "width"
    height_dim: str = "height"

    def __post_init__(self):
        super().__post_init__()

        self.horizontal_dims: List[str] = [
            self.face_dim,
            self.width_dim,
            self.height_dim,
        ]
        self.lat_lon_dims: List[str] = [self.latitude_dim, self.longitude_dim]


@dataclasses.dataclass
class _ChunkingConfig(abc.ABC):
    time_dim: int = 160

    @abc.abstractmethod
    def get_chunks(self, standard_names: StandardNameMapping) -> Dict[str, int]: ...


@dataclasses.dataclass
class ChunkingConfig(_ChunkingConfig):
    latitude_dim: int = -1
    longitude_dim: int = -1

    def get_chunks(self, standard_names: StandardNameMapping) -> Dict[str, int]:
        return {
            standard_names.time_dim: self.time_dim,
            standard_names.longitude_dim: self.longitude_dim,
            standard_names.latitude_dim: self.latitude_dim,
        }


@dataclasses.dataclass
class DLWPChunkingConfig(_ChunkingConfig):
    face_dim: int = -1
    width_dim: int = -1
    height_dim: int = -1

    def get_chunks(self, standard_names: StandardNameMapping) -> Dict[str, int]:
        dlwp_names = standard_names
        if not isinstance(dlwp_names, DLWPNameMapping):
            raise TypeError(
                "Expected DLWPChunkingConfig to be passed type of DLWPNameMapping."
            )
        chunks = {
            dlwp_names.time_dim: self.time_dim,
            dlwp_names.face_dim: self.face_dim,
            dlwp_names.width_dim: self.width_dim,
            dlwp_names.height_dim: self.height_dim,
        }
        return chunks


@dataclasses.dataclass
class DatasetComputationConfig:
    """Configuration of computation details for an FME reference dataset.

    Parameters:
        reference_vertical_coordinate_file: path to netCDF file containing
            vertical coordinate definition for the reference simulation.
        vertical_coarsening_indices: list of tuples defining the ranges of
            reference levels that go into each vertically coarsened layer.
        variable_sources: mapping of zarr store names, e.g. "full_state.zarr",
            to lists of variables to extract from each.
        n_split: number of steps to split the computation over across time.
        roundtrip_fraction_kept: (optional) fraction of spherical harmonics to
            keep in roundtrip transform. Must be between 0 and 1. If omitted,
            the default, no roundtrip transform is applied.
        renaming: (optional) mapping of names in dataset to renamed output
        standard_names: (optional) mapping of standard names to corresponding
            names of variables in the dataset.
        chunking: (optional) mapping of standard dimension names to desired
            output chunk sizes
        time_invariant_dir: (optional) path to directory containing time-invariant data
            This option is used for E3SMv2 dataset.
    """

    reference_vertical_coordinate_file: str
    vertical_coarsening_indices: Sequence[Tuple[int, int]]
    variable_sources: Mapping[str, Sequence[str]]
    n_split: int = 65
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)
    roundtrip_fraction_kept: Optional[float] = None
    standard_names: Union[StandardNameMapping, DLWPNameMapping] = dataclasses.field(
        default_factory=StandardNameMapping
    )
    chunking: Union[ChunkingConfig, DLWPChunkingConfig] = dataclasses.field(
        default_factory=ChunkingConfig
    )
    time_invariant_dir: Optional[str] = None


@dataclasses.dataclass
class DatasetConfig:
    """Dataset provenance for a set of reference simulations.

    Parameters:
        runs: mapping of short names to full paths of reference datasets.
        output_directory: path to place output of computation script.
        dataset_computation: configuration details for dataset
            computation.
        stats_config: configuration to retrieve statistics dataset
    """

    runs: Mapping[str, str]
    data_output_directory: str
    dataset_computation: DatasetComputationConfig
    stats: StatsConfig

    @classmethod
    def from_file(cls, path: str) -> "DatasetConfig":
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


def weighted_mean(da: xr.DataArray, weights: xr.DataArray, dims) -> xr.DataArray:
    """Compute weighted mean of xr.DataArray."""
    return (da * weights).sum(dims) / weights.sum(dims)


def get_dataset_urls(
    config: DatasetComputationConfig, run_directory: str
) -> MutableMapping[str, str]:
    return {k: os.path.join(run_directory, k) for k in config.variable_sources}


def open_datasets(
    config: DatasetComputationConfig, urls: MutableMapping[str, str]
) -> xr.Dataset:
    datasets = []
    for store, names in config.variable_sources.items():
        url = urls[store]
        ds = xr.open_zarr(url)[names]
        datasets.append(ds)
    return xr.merge(datasets, compat="equals")


def get_coarse_ak_bk(
    url: str,
    interface_indices: Sequence[Tuple[int, int]],
    z_dim="xaxis_1",
    time_dim="Time",
) -> xr.Dataset:
    """Return dataset with scalar ak and bk coordinates that define coarse interfaces.

    Args:
        url: path to netCDF file with ak and bk variables in format output by FV3GFS.
        interface_indices: list of tuples of indices of the interfaces in the vertical.
        z_dim: name of dimension along which ak and bk are defined.
        time_dim: name of time dimension.

    Returns:
        xr.Dataset with ak and bk variables as scalars labeled as ak_0, bk_0, etc.

    Note:
        The ak and bk variables will have one more vertical level than the other 3D
        variables since they represent the interfaces between levels.
    """
    with fsspec.open(url) as f:
        vertical_coordinate = xr.open_dataset(f).load()
    # squeeze out the singleton time dimension
    vertical_coordinate = vertical_coordinate.squeeze()
    data = {}
    for i, (start, end) in enumerate(interface_indices):
        data[f"ak_{i}"] = vertical_coordinate["ak"].isel({z_dim: start})
        data[f"bk_{i}"] = vertical_coordinate["bk"].isel({z_dim: start})
        if i == len(interface_indices) - 1:
            data[f"ak_{i + 1}"] = vertical_coordinate["ak"].isel({z_dim: end})
            data[f"bk_{i + 1}"] = vertical_coordinate["bk"].isel({z_dim: end})
    for i in range(len(interface_indices) + 1):
        data[f"ak_{i}"].attrs["units"] = "Pa"
        data[f"bk_{i}"].attrs["units"] = ""  # unitless quantity
        for name in ["ak", "bk"]:
            data[f"{name}_{i}"] = data[f"{name}_{i}"].drop([z_dim, time_dim])
    return xr.Dataset(data)


def compute_ocean_fraction(
    ds: xr.Dataset,
    output_name: str,
    land_fraction_name: str,
    sea_ice_fraction_name: str,
) -> xr.Dataset:
    """Compute latent heat flux, if needed."""
    if output_name in ds.data_vars:
        # if ocean_fraction is already computed, assume that NaNs have been handled
        return ds
    ds[sea_ice_fraction_name] = ds[sea_ice_fraction_name].fillna(0.0)
    ocean_fraction = 1 - ds[sea_ice_fraction_name] - ds[land_fraction_name]
    negative_ocean = xr.where(ocean_fraction < 0, ocean_fraction, 0)
    ocean_fraction -= negative_ocean
    ds["sea_ice_fraction"] += negative_ocean
    ocean_fraction.attrs["units"] = "unitless"
    ocean_fraction.attrs["long_name"] = "fraction of grid cell area occupied by ocean"
    return ds.assign({output_name: ocean_fraction})


def compute_latent_heat_flux(
    ds: xr.Dataset,
    output_name: str,
    evaporation_name: Optional[str] = None,
) -> xr.Dataset:
    """Compute latent heat flux, if needed."""
    if output_name in ds.data_vars:
        return ds
    assert (
        evaporation_name is not None
    ), f"{output_name} not found in ds, evaporation_name must be provided."
    latent_heat_flux = ds[evaporation_name] * LATENT_HEAT_OF_VAPORIZATION
    latent_heat_flux.attrs["units"] = "W/m^2"
    latent_heat_flux.attrs["long_name"] = "Latent heat flux"
    return ds.assign({output_name: latent_heat_flux}).drop(evaporation_name)


def compute_specific_total_water(
    ds: xr.Dataset, water_condensate_names: Sequence[str], output_name: str
) -> xr.Dataset:
    """Compute specific total water from individual water species."""
    specific_total_water: xr.DataArray = sum(
        [ds[name] for name in water_condensate_names]
    )
    specific_total_water.attrs["units"] = "kg/kg"
    specific_total_water.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: specific_total_water})


def compute_pressure_thickness(
    ds: xr.Dataset,
    vertical_coordinate_file: str,
    vertical_dim_name: str,
    surface_pressure_name: str,
    output_name: str,
    z_dim: str = "xaxis_1",
):
    if output_name in ds.data_vars:
        return ds

    with fsspec.open(vertical_coordinate_file) as f:
        vertical_coordinate = xr.open_dataset(f).load()
    # squeeze out the singleton time dimension
    vertical_coord = vertical_coordinate.squeeze(drop=True)

    sfc_pressure = ds[surface_pressure_name].expand_dims(
        {z_dim: vertical_coord[z_dim]}, axis=3
    )
    phalf = sfc_pressure * vertical_coord["bk"] + vertical_coord["ak"]

    thickness = (
        phalf.diff(dim=z_dim)
        .rename({z_dim: vertical_dim_name})
        .rename(output_name)
        .assign_coords(
            {vertical_dim_name: (vertical_dim_name, ds[vertical_dim_name].values)}
        )
    )

    thickness.attrs["units"] = "Pa"
    thickness.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: thickness})


def compute_vertical_coarsening(
    ds: xr.Dataset,
    vertically_resolved_names: Sequence[str],
    interface_indices: Sequence[Tuple[int, int]],
    dim: str,
    pressure_thickness_name: str,
) -> xr.Dataset:
    """Compute vertical coarsening of 3D variables by mass-weighted mean. Outputs are
    saved as new variables in the dataset with the name '{name}_{i}' where i is the
    new coarse vertical level index."""
    coarsened_arrays = {}
    for i, (start, end) in enumerate(interface_indices):
        pressure_thickness = ds[pressure_thickness_name].isel({dim: slice(start, end)})
        for name in vertically_resolved_names:
            array_slice = ds[name].isel({dim: slice(start, end)})
            coarsened_da = weighted_mean(array_slice, pressure_thickness, dim)
            current_long_name = array_slice.long_name
            coarsened_da.attrs["long_name"] = current_long_name + f" level-{i}"
            coarsened_arrays[f"{name}_{i}"] = coarsened_da
    return ds.assign(coarsened_arrays)


def compute_tendencies(
    ds: xr.Dataset, time_derivative_names: Sequence[str], dim: str
) -> xr.Dataset:
    """Compute backward difference over time dimension. The tendency variables
    will be NaNs for the first timestep in the output dataset."""
    # this code does not assume that all time steps are equally spaced
    timestep_seconds = (ds[dim].diff(dim) / np.timedelta64(1, "s")).astype("float32")
    tendencies = {}
    for name in time_derivative_names:
        tendency = ds[name].diff(dim) / timestep_seconds
        tendency.attrs["units"] = f"{ds[name].units}/s"
        tendency.attrs["long_name"] = f"time derivative of {ds[name].long_name}"
        tendencies[f"tendency_of_{name}"] = tendency
    return ds.assign(tendencies)


def compute_column_advective_moisture_tendency(
    ds: xr.Dataset,
    pwat_tendency: str,
    latent_heat_flux: str,
    precip: str,
    latent_heat_of_vaporization: float,
) -> xr.Dataset:
    evaporation = ds[latent_heat_flux] / latent_heat_of_vaporization
    advective_tendency = ds[pwat_tendency] - evaporation + ds[precip]
    long_name = "tendency of total water path due to advection"
    advective_tendency.attrs["long_name"] = long_name
    return ds.assign({f"{pwat_tendency}_due_to_advection": advective_tendency})


def compute_column_moisture_integral(
    ds: xr.Dataset,
    input_name: str,
    output_name: str,
    pressure_thickness_name: str,
    dim: str,
) -> xr.Dataset:
    """Compute the column integral of a mass mixing ratio."""
    column_integral = (ds[input_name] * ds[pressure_thickness_name]).sum(dim) / GRAVITY
    column_integral.attrs["units"] = "kg/m^2"
    column_integral.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: column_integral})


def assert_column_integral_of_moisture_is_conserved(
    ds: xr.Dataset, precipitable_water_path_name: str, total_water_path_name: str
) -> None:
    """Assert that the column integral of 'specific_total_water' is close to the
    precipitable water path that is computed online on native grid by FV3GFS. Note
    that there are pretty large difference (rtol=1e-1) which is likely due to the
    fregrid tool doing area-weighted average instead of mass-weighted."""
    expected_pwat = ds[precipitable_water_path_name]
    integrated_pwat = ds[total_water_path_name]
    print("Mean absolute difference between expected and integrated pwat [kg/m^2]:")
    print(np.abs(expected_pwat - integrated_pwat).mean().values)
    xr.testing.assert_allclose(integrated_pwat, expected_pwat, rtol=1e-1, atol=1e-3)


def assert_global_dry_air_mass_conservation(
    ds: xr.Dataset,
    dims: List[str],
    surface_pressure_name: str,
    total_water_path_name: str,
    latitude_dim: str,
    time_dim: str,
) -> None:
    """Assert that the tendency of global average surface pressure (due to dry air
    only) is close to zero. I.e. dry air mass is conserved."""
    column_dry_air_mass = (
        ds[surface_pressure_name] - ds[total_water_path_name] * GRAVITY
    )
    if latitude_dim in dims:
        weights = np.cos(np.deg2rad(ds[latitude_dim]))
        global_dry_air_mass = column_dry_air_mass.weighted(weights).mean(dim=dims)
    else:
        global_dry_air_mass = column_dry_air_mass.mean(dim=dims)

    global_dry_air_mass_tendency = global_dry_air_mass.diff(time_dim)
    print("Mean absolute global dry air pressure tendency [Pa]:")
    print(np.abs(global_dry_air_mass_tendency).mean().values)
    xr.testing.assert_allclose(
        global_dry_air_mass_tendency,
        xr.zeros_like(global_dry_air_mass_tendency),
        atol=1e-3,
    )


def assert_global_moisture_conservation(
    ds: xr.Dataset,
    dims: List[str],
    latitude_dim: str,
    total_water_path_name: str,
    latent_heat_flux_name: str,
    latent_heat_of_vaporization: float,
    precip_rate_name: str,
    time_dim: str,
) -> None:
    """Assert that the tendency of global average column integrated moisture is equal
    to the global average flux of moisture through the surface."""
    integrated_pwat = ds[total_water_path_name]
    weights = np.cos(np.deg2rad(ds[latitude_dim]))
    global_moisture = integrated_pwat.weighted(weights).mean(dim=dims)
    timestep_seconds = ds[time_dim].diff(time_dim) / np.timedelta64(1, "s")
    actual_global_moisture_tendency = global_moisture.diff(time_dim) / timestep_seconds
    evap_minus_precip = (
        ds[latent_heat_flux_name] / latent_heat_of_vaporization - ds[precip_rate_name]
    )
    expected_global_moisture_tendency = (
        evap_minus_precip.weighted(weights).mean(dim=dims).isel(time=slice(1, None))
    )
    print("Mean absolute global moisture non-conservative source [kg/m^2/s]:")
    diff = actual_global_moisture_tendency - expected_global_moisture_tendency
    print(np.abs(diff).mean().values)
    xr.testing.assert_allclose(
        expected_global_moisture_tendency, actual_global_moisture_tendency
    )


def construct_lazy_dataset(
    config: DatasetComputationConfig, run_directory: str
) -> xr.Dataset:
    standard_names = config.standard_names
    urls = get_dataset_urls(config, run_directory)
    ds = open_datasets(config, urls)
    for var in ds:
        del ds[var].encoding["chunks"]
        del ds[var].encoding["preferred_chunks"]
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    if config.roundtrip_fraction_kept is not None:
        ds = xtorch_harmonics.roundtrip_filter(
            ds,
            lat_dim=standard_names.latitude_dim,
            lon_dim=standard_names.longitude_dim,
            fraction_modes_kept=config.roundtrip_fraction_kept,
        )
    ds = compute_ocean_fraction(
        ds,
        output_name=standard_names.ocean_fraction,
        land_fraction_name=standard_names.land_fraction,
        sea_ice_fraction_name=standard_names.sea_ice_fraction,
    )
    ds = compute_latent_heat_flux(
        ds,
        output_name=standard_names.latent_heat_flux,
        evaporation_name=standard_names.surface_evaporation_rate,
    )
    ds = compute_specific_total_water(
        ds,
        water_condensate_names=standard_names.water_species,
        output_name=standard_names.specific_total_water,
    )
    ds = compute_pressure_thickness(
        ds,
        vertical_coordinate_file=config.reference_vertical_coordinate_file,
        vertical_dim_name=standard_names.vertical_dim,
        surface_pressure_name=standard_names.surface_pressure,
        output_name=standard_names.pressure_thickness,
    )
    ds = compute_vertical_coarsening(
        ds,
        vertically_resolved_names=standard_names.vertically_resolved,
        interface_indices=config.vertical_coarsening_indices,
        dim=standard_names.vertical_dim,
        pressure_thickness_name=standard_names.pressure_thickness,
    )
    ds = compute_column_moisture_integral(
        ds,
        input_name=standard_names.specific_total_water,
        output_name=standard_names.total_water_path,
        pressure_thickness_name=standard_names.pressure_thickness,
        dim=standard_names.vertical_dim,
    )
    ds = compute_tendencies(
        ds,
        time_derivative_names=standard_names.time_derivative_names,
        dim=standard_names.time_dim,
    )
    ds = compute_column_advective_moisture_tendency(
        ds,
        pwat_tendency=standard_names.pwat_tendency,
        latent_heat_flux=standard_names.latent_heat_flux,
        precip=standard_names.precip_rate,
        latent_heat_of_vaporization=LATENT_HEAT_OF_VAPORIZATION,
    )
    ak_bk_ds = get_coarse_ak_bk(
        config.reference_vertical_coordinate_file,
        config.vertical_coarsening_indices,
    )
    ds = xr.merge([ds, ak_bk_ds])
    chunks = config.chunking.get_chunks(standard_names)
    ds = ds.chunk(chunks)
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts/data_process"
        "/compute_dataset_fv3gfs.py"
        f" script, using following input zarrs: {urls.values()}."
    )
    ds.attrs["vertical_coordinate"] = (
        "The pressure at level interfaces can by computed as "
        "p_i = ak_i + bk_i * PRESsfc, where PRESsfc is the surface pressure and the "
        "p_i pressure corresponds to the interface at the top of the i'th finite "
        "volume layer, counting down from the top of atmosphere."
    )
    ds = ds.rename(config.renaming)
    return ds


@click.command()
@click.option("--config", help="Path to dataset configuration YAML file.")
@click.option("--run-directory", help="Path to reference run directory.")
@click.option("--output-store", help="Path to output zarr store.")
@click.option("--debug", is_flag=True, help="Print metadata instead of writing output.")
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
@click.option("--check-conservation", is_flag=True, help="Check conservation.")
def main(
    config,
    run_directory,
    output_store,
    debug,
    subsample,
    check_conservation,
):
    config = DatasetConfig.from_file(config).dataset_computation
    print(f"--run-directory is {run_directory}")
    print(f"--output-store is {output_store}")
    standard_names = config.standard_names
    xr.set_options(keep_attrs=True)
    ds = construct_lazy_dataset(config, run_directory)
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    if check_conservation:
        assert_column_integral_of_moisture_is_conserved(
            ds,
            precipitable_water_path_name=standard_names.precipitable_water_path,
            total_water_path_name=standard_names.total_water_path,
        )
        assert_global_dry_air_mass_conservation(
            ds,
            dims=standard_names.horizontal_dims,
            surface_pressure_name=standard_names.surface_pressure,
            total_water_path_name=standard_names.total_water_path,
            latitude_dim=standard_names.latitude_dim,
            time_dim=standard_names.time_dim,
        )
        assert_global_moisture_conservation(
            ds,
            dims=standard_names.horizontal_dims,
            latitude_dim=standard_names.latitude_dim,
            total_water_path_name=standard_names.total_water_path,
            latent_heat_flux_name=standard_names.latent_heat_flux,
            latent_heat_of_vaporization=LATENT_HEAT_OF_VAPORIZATION,
            precip_rate_name=standard_names.precip_rate,
            time_dim=standard_names.time_dim,
        )
    ds = ds.drop(standard_names.dropped_variables)
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")
    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        ds.partition.initialize_store(output_store)
        for i in range(config.n_split):
            print(f"Writing segment {i + 1} / {config.n_split}")
            with ProgressBar():
                ds.partition.write(
                    output_store,
                    config.n_split,
                    [config.standard_names.time_dim],
                    i,
                    collect_variable_writes=True,
                )


if __name__ == "__main__":
    main()
