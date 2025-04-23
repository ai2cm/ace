# This script is used to compute a training dataset from the "raw"
# FV3GFS data stored in zarr form on GCS.

# The resulting dataset is about 194GB (the input is about 2.5TB). Running this script
# on my 8-CPU VM takes about 2.5 hours. See "compute_dataset_fv3gfs_argo_workflow.yaml"
# for a workflow which parallelizes this script across the 11-member ensemble and runs
# it on our GKE cluster.

import abc
import dataclasses
import logging
import os
import sys
import time
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import click
import dacite
import fsspec
import numpy as np
import xarray as xr
import yaml

try:
    from xtorch_harmonics import roundtrip_filter
except ModuleNotFoundError:

    def roundtrip_filter(*args, **kwargs):
        raise ModuleNotFoundError("xtorch_harmonics")


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
SURFACE_FROZEN_PRECIPITATION_NAME = "total_frozen_precipitation_rate"


@dataclasses.dataclass
class StandardDimMapping:
    longitude_dim: str = "grid_xt"
    latitude_dim: str = "grid_yt"
    time_dim: str = "time"


@dataclasses.dataclass
class StandardNameMapping(StandardDimMapping):
    vertical_dim: str = "pfull"
    vertical_interface_dim: str = "phalf"
    surface_pressure: str = "PRESsfc"
    latent_heat_flux: str = "LHTFLsfc"
    precip_rate: str = "PRATEsfc"
    surface_snow_rate: str = "SNOWsfc"
    surface_ice_rate: str = "ICEsfc"
    surface_graupel_rate: str = "GRAUPELsfc"
    total_frozen_precip_rate: str = "total_frozen_precipitation_rate"
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
    vertical_dim_land: str = "zfull_soil"
    height_thickness: str = "height_thickness_of_land_layer"
    total_moisture_content_of_soil_layer: str = "total_moisture_content_of_soil_layer"
    hybrid_level_coeffs: List[str] = dataclasses.field(default_factory=list)
    additional_vertically_resolved_names: List[str] = dataclasses.field(
        default_factory=list
    )
    land_names_to_vertically_coarsen_by_height_weighting: List[str] = dataclasses.field(
        default_factory=list
    )
    land_names_to_vertically_coarsen_by_sum: List[str] = dataclasses.field(
        default_factory=list
    )

    def __post_init__(self):
        self.horizontal_dims: List[str] = [self.longitude_dim, self.latitude_dim]

        self.specific_total_water = SPECIFIC_TOTAL_WATER
        self.total_water_path = TOTAL_WATER_PATH
        self.total_frozen_precip_rate_output_name = SURFACE_FROZEN_PRECIPITATION_NAME
        self.pwat_tendency = f"tendency_of_{self.total_water_path}"
        self.time_derivative_names = [self.total_water_path]

        self.vertically_resolved: List[str] = [
            self.specific_total_water,
            self.air_temperature,
            self.northward_wind,
            self.eastward_wind,
        ] + self.additional_vertically_resolved_names

        self.vertically_resolved_names_land: List[str] = (
            self.land_names_to_vertically_coarsen_by_height_weighting
            + self.land_names_to_vertically_coarsen_by_sum
        )

        # variables to drop after all derived variables are computed
        self.dropped_variables: List[str] = (
            self.water_species
            + self.vertically_resolved
            + [self.pressure_thickness, self.vertical_dim]
            + self.vertically_resolved_names_land
        )
        for name in [
            self.precipitable_water_path,
            self.surface_graupel_rate,
            self.surface_ice_rate,
            self.surface_snow_rate,
        ]:
            if name.lower() != "none":
                self.dropped_variables.append(name)
        if self.vertically_resolved_names_land:
            self.dropped_variables.append(self.vertical_dim_land)

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

    @property
    def frozen_precipitation_species(self) -> List[str]:
        if self.total_frozen_precip_rate.lower() != "none":
            # if total frozen precip rate is available, just use that
            return [self.total_frozen_precip_rate]
        else:
            # return all frozen precip species
            return [
                item
                for item in [
                    self.surface_graupel_rate,
                    self.surface_ice_rate,
                    self.surface_snow_rate,
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
    time_dim: int = 1

    @abc.abstractmethod
    def get_chunks(self, standard_names: StandardDimMapping) -> Dict[str, int]: ...


@dataclasses.dataclass
class ChunkingConfig(_ChunkingConfig):
    latitude_dim: int = -1
    longitude_dim: int = -1

    def get_chunks(self, standard_names: StandardDimMapping) -> Dict[str, int]:
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

    def get_chunks(self, standard_names: StandardDimMapping) -> Dict[str, int]:
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
        validate_vertical_coarsening_indices: (optional) whether to check that
            vertical coarsening indices span the full atmosphere and do not
            overlap (default True).
        n_split: number of steps to split the computation over across time.
        roundtrip_fraction_kept: (optional) fraction of spherical harmonics to
            keep in roundtrip transform. Must be between 0 and 1. If omitted,
            the default, no roundtrip transform is applied.
        renaming: (optional) mapping of names in dataset to renamed output
        standard_names: (optional) mapping of standard names to corresponding
            names of variables in the dataset.
        chunking: (optional) mapping of standard dimension names to desired
            output inner chunk sizes. Defaults to a chunk size of 1 along
            the time dimension.
        sharding: (optional) mapping of standard dimension names to desired
            output shard sizes. Defaults to a shard size of 360 along the time
            dimension. If None, then an unsharded zarr store will be written
            with chunks as specified in ``chunking``.
        time_invariant_dir: (optional) path to directory containing time-invariant data
            This option is used for E3SMv2 dataset.
        vertical_coarsening_indices_land: (optional) list of tuples defining the ranges
            of reference levels that go into each vertically coarsened layer for the
            land model variables.
        validate_vertical_coarsening_indices_land: (optional) whether to check that
            land vertical coarsening indices span the full depth of the land model
            and do not overlap (default True).
        reference_vertical_coordinate_file_land: (optional) path to netCDF file
            containing vertical coordinate definition for the land model of the
            reference simulation.
        mask_soil_moisture: (optional) whether to mask soil moisture content using soil
            temperature. This is useful for CM4 dataset, where soil moisture content is
            zero instead of NaN over the oceans.
    """

    reference_vertical_coordinate_file: str
    vertical_coarsening_indices: Sequence[Tuple[int, int]]
    variable_sources: Mapping[str, Sequence[str]]
    validate_vertical_coarsening_indices: bool = True
    n_split: int = 65
    renaming: Mapping[str, str] = dataclasses.field(default_factory=dict)
    roundtrip_fraction_kept: Optional[float] = None
    standard_names: Union[StandardNameMapping, DLWPNameMapping] = dataclasses.field(
        default_factory=StandardNameMapping
    )
    chunking: Union[ChunkingConfig, DLWPChunkingConfig] = dataclasses.field(
        default_factory=lambda: ChunkingConfig(time_dim=1)
    )
    sharding: Optional[Union[ChunkingConfig, DLWPChunkingConfig]] = dataclasses.field(
        default_factory=lambda: ChunkingConfig(time_dim=360)
    )
    time_invariant_dir: Optional[str] = None
    vertical_coarsening_indices_land: Optional[Sequence[Tuple[int, int]]] = None
    validate_vertical_coarsening_indices_land: bool = True
    reference_vertical_coordinate_file_land: Optional[str] = None
    mask_soil_moisture: bool = False


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
    return (da * weights).sum(dims, skipna=False) / weights.sum(dims)


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
        ds = xr.open_zarr(url, decode_timedelta=True)[names]
        datasets.append(ds)
    return xr.merge(datasets, compat="equals")


def get_coarse_ak_bk(
    url: str,
    interface_indices: Sequence[Tuple[int, int]],
    z_dim="xaxis_1",
    time_dim="Time",
    dtype=np.float32,
) -> xr.Dataset:
    """Return dataset with scalar ak and bk coordinates that define coarse interfaces.

    Args:
        url: path to netCDF file with ak and bk variables in format output by FV3GFS.
        interface_indices: list of tuples of indices of the interfaces in the vertical.
        z_dim: name of dimension along which ak and bk are defined.
        time_dim: name of time dimension.
        dtype: data type (e.g., np.float32) for ak and bk

    Returns:
        xr.Dataset with ak and bk variables as scalars labeled as ak_0, bk_0, etc.

    Note:
        The ak and bk variables will have one more vertical level than the other 3D
        variables since they represent the interfaces between levels.
    """
    with fsspec.open(url) as f:
        vertical_coordinate = xr.open_dataset(f, decode_timedelta=False).load()

    vertical_coordinate = vertical_coordinate.astype(dtype)

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
            data[f"{name}_{i}"] = data[f"{name}_{i}"].drop_vars([z_dim, time_dim])
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
    return ds.assign({output_name: latent_heat_flux}).drop_vars(evaporation_name)


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


def compute_frozen_precipitation_rate(
    ds: xr.Dataset, frozen_precip_names: Sequence[str], output_name: str
) -> xr.Dataset:
    """Compute the total surface frozen precipitation rate."""
    frozen_precip: xr.DataArray = sum([ds[name] for name in frozen_precip_names])
    frozen_precip.attrs["units"] = ds[frozen_precip_names[0]].units
    frozen_precip.attrs["long_name"] = "Total surface frozen precipitation rate"
    return ds.assign({output_name: frozen_precip})


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
        vertical_coordinate = xr.open_dataset(f, decode_timedelta=False).load()

    vertical_coordinate = vertical_coordinate.astype(ds[surface_pressure_name].dtype)

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


def validate_vertical_coarsening_indices(
    vertical_dim_size: int,
    interface_indices: Sequence[Tuple[int, int]],
    component: str,
    control_flag: str,
) -> None:
    """Check that vertical coarsening indices span the full vertical dimension of
    the model component and do not overlap.

    Args:
        vertical_dim_size: size of the vertical dimension in the input dataset.
        interface_indices: provided vertical coarsening indices.
        component: model component the vertical dimension corresponds to (only
            relevant for providing a clearer error message).
        control_flag: relevant control flag on DatasetComputationConfig to disable
            this check if desired (only relevant for providing a clearer error
            message).
    """
    expected_covered_indices = list(range(vertical_dim_size))
    covered_indices = []
    for start, end in interface_indices:
        covered_indices.extend(list(range(start, end)))

    if covered_indices != expected_covered_indices:
        raise ValueError(
            f"Provided {component} vertical coarsening indices {interface_indices!r} "
            f"do not exactly span all {vertical_dim_size} vertical levels of the "
            f"input dataset and/or consist of overlapping ranges of levels. If your "
            f"intent is to use incomplete or overlapping levels, you may disable this "
            f"check by setting {control_flag} in DatasetComputationConfig to False. "
            f"If this is not your intent, double check that you are using the proper "
            f"coarsening indices and, if relevant, the vertical coordinate reference "
            f"file for this reference model component configuration."
        )


def compute_vertical_coarsening(
    ds: xr.Dataset,
    vertically_resolved_names: Sequence[str],
    interface_indices: Sequence[Tuple[int, int]],
    dim: str,
    pressure_thickness_name: str,
    validate_indices: bool,
) -> xr.Dataset:
    """Compute vertical coarsening of 3D variables by mass-weighted mean. Outputs are
    saved as new variables in the dataset with the name '{name}_{i}' where i is the
    new coarse vertical level index."""

    if validate_indices:
        validate_vertical_coarsening_indices(
            ds.sizes[dim],
            interface_indices,
            "atmosphere",
            "validate_vertical_coarsening_indices",
        )

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


def compute_vertical_coarsening_land(
    ds: xr.Dataset,
    vertically_resolved_names: Sequence[str],
    interface_indices: Optional[Sequence[Tuple[int, int]]],
    vertical_coordinate_file: Optional[str],
    dim: str,
    height_thickness_name: str,
    summed_variables: Sequence[str],
    validate_indices: bool,
    mask_soil_moisture: bool = False,
) -> xr.Dataset:
    """Compute vertical coarsening of 3D land variables by height-weighted mean or
    unweighted sum. Outputs are saved as new variables in the dataset with the
    name '{name}_{i}' where i is the new coarse vertical level index. Variables are
    coarsened by a height-weighted mean by default. Variables listed in
    summed_variables are coarsened using an unweighted sum. This is useful, for
    example, for the total_moisture_content_of_soil_layer in LM4, which has units
    of total moisture (kg / m^2).
    """

    assert interface_indices is not None, (
        "Land variables for coarsening are provided, but there"
        "are not corresponding coarsening indices"
    )

    assert vertical_coordinate_file is not None, (
        "Land variables for coarsening are provided, but there"
        "is not a corresponding reference file"
    )

    if not vertically_resolved_names:
        return ds

    if validate_indices:
        validate_vertical_coarsening_indices(
            ds.sizes[dim],
            interface_indices,
            "land",
            "validate_vertical_coarsening_indices_land",
        )

    with fsspec.open(vertical_coordinate_file) as f:
        thickness = xr.open_dataset(f, decode_timedelta=False).load()

    coarsened_arrays = {}

    if mask_soil_moisture and (
        "total_moisture_content_of_soil_layer" in vertically_resolved_names
        and "temperature_of_soil_layer" in vertically_resolved_names
    ):
        # Mask out soil moisture content when temperature is NaN
        logging.info("Masking soil moisture content using temperature")
        ds["total_moisture_content_of_soil_layer"] = ds[
            "total_moisture_content_of_soil_layer"
        ].where(ds.temperature_of_soil_layer.notnull())

    for i, (start, end) in enumerate(interface_indices):
        height_thickness = thickness[height_thickness_name].isel(
            {dim: slice(start, end)}
        )
        for name in vertically_resolved_names:
            array_slice = ds[name].isel({dim: slice(start, end)})

            # some land variables are total quantity in layer so just need to be summed
            if name in summed_variables:
                coarsened_da = array_slice.sum(dim, skipna=False)
            else:
                coarsened_da = weighted_mean(
                    array_slice, height_thickness.astype(array_slice.dtype), dim
                )

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
    logging.info(
        f"Mean absolute difference between expected and integrated pwat [kg/m^2]: "
        f"{np.abs(expected_pwat - integrated_pwat).mean().values}"
    )
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
    logging.info(
        f"Mean absolute global dry air pressure tendency [Pa]: "
        f"{np.abs(global_dry_air_mass_tendency).mean().values}"
    )
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
    diff = actual_global_moisture_tendency - expected_global_moisture_tendency
    logging.info(
        f"Mean absolute global moisture non-conservative source [kg/m^2/s]: "
        f"{np.abs(diff).mean().values}"
    )
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
    logging.info(f"Input dataset size is {ds.nbytes / 1e9} GB")

    if config.roundtrip_fraction_kept is not None:
        ds = roundtrip_filter(
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
    ds = compute_frozen_precipitation_rate(
        ds,
        frozen_precip_names=standard_names.frozen_precipitation_species,
        output_name=standard_names.total_frozen_precip_rate_output_name,
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
        validate_indices=config.validate_vertical_coarsening_indices,
    )

    if standard_names.vertically_resolved_names_land:
        ds = compute_vertical_coarsening_land(
            ds,
            vertically_resolved_names=standard_names.vertically_resolved_names_land,
            interface_indices=config.vertical_coarsening_indices_land,
            vertical_coordinate_file=config.reference_vertical_coordinate_file_land,
            dim=standard_names.vertical_dim_land,
            height_thickness_name=standard_names.height_thickness,
            summed_variables=standard_names.land_names_to_vertically_coarsen_by_sum,
            validate_indices=config.validate_vertical_coarsening_indices_land,
            mask_soil_moisture=config.mask_soil_moisture,
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
        dtype=ds[standard_names.surface_pressure].dtype,
    )
    ds = xr.merge([ds, ak_bk_ds])

    if config.sharding is None:
        outer_chunks = config.chunking.get_chunks(standard_names)
    else:
        outer_chunks = config.sharding.get_chunks(standard_names)

    ds = ds.chunk(outer_chunks)

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


def clear_compressors_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Clear "compressors" encoding of the Dataset.

    This ensures that we use zarr's default zstd encoding when writing out the
    Dataset. It also helps avoid errors resulting from lingering zarr v2
    compression encoding parameters that are not compatible with zarr v3.

    Args:
        ds: input xr.Dataset to remove "compressors" encoding from.

    Returns:
        xr.Dataset with the "compressors" key in the encoding dictionary of
        each variable removed, if it exists.
    """
    ds = ds.copy(deep=False)
    for variable in {**ds.coords, **ds.data_vars}.values():
        variable.encoding.pop("compressors", None)
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
    # Import distributed and xpartition here to allow testing functions in
    # the fme environment.
    import distributed
    import xpartition  # noqa: F401

    logging.basicConfig(level=logging.INFO)
    distributed.Client(n_workers=16)

    config = DatasetConfig.from_file(config).dataset_computation
    logging.info(f"--run-directory is {run_directory}")
    logging.info(f"--output-store is {output_store}")
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
    ds = ds.drop_vars(standard_names.dropped_variables)

    if config.sharding is None:
        inner_chunks = None
    else:
        inner_chunks = config.chunking.get_chunks(standard_names)

    ds = clear_compressors_encoding(ds)

    logging.info(f"Output dataset size is {ds.nbytes / 1e9} GB")
    if debug:
        with xr.set_options(display_max_rows=500):
            logging.info(ds)
    else:
        ds.partition.initialize_store(output_store, inner_chunks=inner_chunks)
        for i in range(config.n_split):
            segment_number = f"{i + 1} / {config.n_split}"
            logging.info(f"Writing segment {segment_number}")
            segment_time = time.time()
            ds.partition.write(
                output_store,
                config.n_split,
                [config.standard_names.time_dim],
                i,
                collect_variable_writes=True,
            )
            segment_time = time.time() - segment_time
            logging.info(f"Segment {segment_number} time: {segment_time:0.2f} seconds")


if __name__ == "__main__":
    main()
