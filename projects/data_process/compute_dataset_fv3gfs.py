# This script is used to compute a training dataset from the "raw"
# FV3GFS data stored in zarr form on GCS.

# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

# The resulting dataset is about 238GB (the input is about 2.7TB). Running this script
# on my 8-CPU VM takes about 2.5 hours. We could write some code to parallelize the job
# across multiple workers, but it doesn't seem worth the trouble right now.

import click
from dask.diagnostics import ProgressBar
import fsspec
from typing import List, MutableMapping, Sequence, Tuple
import numpy as np
import xarray as xr
import xpartition  # noqa: 401

FLUXES_2D = "fluxes_2d"
FOURCASTNET_VANILLA = "fourcastnet_vanilla"
FULL_STATE = "full_state"
TENDENCIES_3D = "tendencies_3d"

OUTPUT_URL = "gs://vcm-ml-intermediate/2023-05-10-vertically-resolved-1deg-fme-dataset.zarr"  # noqa: 501

# constants are defined as in FV3GFS model
# https://github.com/ai2cm/fv3gfs-fortran/blob/master/FMS/constants/constants.F90
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg
GRAVITY = 9.80665  # m/s^2

SPECIFIC_TOTAL_WATER = "specific_total_water"
# following name is for the water path computed online by FV3GFS with the native grid
PRECIPITABLE_WATER_PATH = "precipitable_water_path"
# following name is for the water path computed by vertically integrating the
# lat-lon 3D variables. This is the water path we want to use for training. It is not
# necessarily exactly equal to the precipitable water path computed online by FV3GFS.
TOTAL_WATER_PATH = "total_water_path"

PRESSURE_THICKNESS = "pressure_thickness_of_atmospheric_layer"
SURFACE_PRESSURE = "PRESsfc"
LATENT_HEAT_FLUX = "LHTFLsfc"
PRECIP_RATE = "PRATEsfc"
TIME_DIM = "time"
HORIZONTAL_DIMS = ["grid_xt", "grid_yt"]
LATITUDE_DIM = "grid_yt"
VERTICAL_DIM = "pfull"

CHUNKS = {"time": 160, "grid_yt": 180, "grid_xt": 360}

# these are assumed to all have the same coordinates
DATASET_URLS = {
    FLUXES_2D: "gs://vcm-ml-raw-flexible-retention/2023-04-13-11-year-C96-FME-reference/regridded-zarrs/fluxes_2d.zarr",  # noqa: 501
    FOURCASTNET_VANILLA: "gs://vcm-ml-raw-flexible-retention/2023-04-13-11-year-C96-FME-reference/regridded-zarrs/fourcastnet_vanilla.zarr",  # noqa: 501
    FULL_STATE: "gs://vcm-ml-raw-flexible-retention/2023-04-13-11-year-C96-FME-reference/regridded-zarrs/full_state.zarr",  # noqa: 501
    TENDENCIES_3D: "gs://vcm-ml-raw-flexible-retention/2023-04-13-11-year-C96-FME-reference/regridded-zarrs/tendencies_3d.zarr",  # noqa: 501
}

VERTICAL_COORDINATE_URL = "gs://vcm-ml-raw-flexible-retention/2023-04-13-11-year-C96-FME-reference/vertical-coordinate-file/fv_core.res.nc"  # noqa: 501

# the variables we need from each of the input zarrs
INPUT_VARIABLE_NAMES = {
    FLUXES_2D: [
        PRECIP_RATE,
        LATENT_HEAT_FLUX,
        "SHTFLsfc",
        "DLWRFsfc",
        "DSWRFsfc",
        "DSWRFtoa",
        "ULWRFsfc",
        "ULWRFtoa",
        "USWRFsfc",
        "USWRFtoa",
        PRECIPITABLE_WATER_PATH,
    ],
    FOURCASTNET_VANILLA: [
        "PRESsfc",
    ],
    FULL_STATE: [
        "surface_temperature",
        "air_temperature",
        "specific_humidity",
        "cloud_water_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "graupel_mixing_ratio",
        "rain_mixing_ratio",
        "snow_mixing_ratio",
        "northward_wind",
        "eastward_wind",
        PRESSURE_THICKNESS,
        "land_sea_mask",
    ],
}

WATER_SPECIES_NAMES = [
    "specific_humidity",
    "cloud_water_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "graupel_mixing_ratio",
    "rain_mixing_ratio",
    "snow_mixing_ratio",
]

VERTICALLY_RESOLVED_NAMES = [
    SPECIFIC_TOTAL_WATER,
    "air_temperature",
    "northward_wind",
    "eastward_wind",
]

TIME_DERIVATIVE_NAMES = [TOTAL_WATER_PATH]

# these indices refer to the vertical interfaces in the input data
# They were computed in this notebook:
# https://github.com/ai2cm/explore/blob/master/oliwm/2023-04-16-fme-analysis/2023-05-03-vertical-coordinate-example.ipynb  # noqa: 501
VERTICAL_LEVEL_INTERFACES = [
    (0, 18),
    (18, 26),
    (26, 31),
    (31, 36),
    (36, 41),
    (41, 47),
    (47, 53),
    (53, 63),
]

# variables to drop after all derived variables are computed
DROP_VARIABLE_NAMES = [
    "air_temperature",
    "specific_humidity",
    "cloud_water_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "graupel_mixing_ratio",
    "rain_mixing_ratio",
    "snow_mixing_ratio",
    "northward_wind",
    "eastward_wind",
    SPECIFIC_TOTAL_WATER,
    PRESSURE_THICKNESS,
    VERTICAL_DIM,
    PRECIPITABLE_WATER_PATH,
]


def weighted_mean(da: xr.DataArray, weights: xr.DataArray, dims) -> xr.DataArray:
    """Compute weighted mean of xr.DataArray."""
    return (da * weights).sum(dims) / weights.sum(dims)


def open_datasets(
    dataset_urls: MutableMapping[str, str],
) -> MutableMapping[str, xr.Dataset]:
    """Open datasets from zarr urls."""
    return {category: xr.open_zarr(url) for category, url in dataset_urls.items()}


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


def merge_inputs(
    input_variable_names: MutableMapping[str, List[str]],
    datasets: MutableMapping[str, xr.Dataset],
) -> xr.Dataset:
    """Merge input variables from multiple zarrs into a single dataset."""
    to_be_merged = []
    for category, variables in input_variable_names.items():
        to_be_merged.append(datasets[category][variables])
    return xr.merge(to_be_merged, compat="equals")


def compute_specific_total_water(
    ds: xr.Dataset,
    water_condensate_names: Sequence[str],
    output_name: str = SPECIFIC_TOTAL_WATER,
) -> xr.Dataset:
    """Compute specific total water from individual water species."""
    specific_total_water = sum([ds[name] for name in water_condensate_names])
    specific_total_water.attrs["units"] = "kg/kg"
    specific_total_water.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: specific_total_water})


def compute_vertical_coarsening(
    ds: xr.Dataset,
    vertically_resolved_names: Sequence[str],
    interface_indices: Sequence[Tuple[int, int]],
    dim: str = VERTICAL_DIM,
    pressure_thickness_name: str = PRESSURE_THICKNESS,
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
    ds: xr.Dataset, time_derivative_names: Sequence[str], dim: str = TIME_DIM
) -> xr.Dataset:
    """Compute backward difference over time dimension. This will result
    in the output dataset having one fewer time steps than the input dataset."""
    # this code does not assume that all time steps are equally spaced
    timestep_seconds = (ds[dim].diff(dim) / np.timedelta64(1, "s")).astype("float32")
    tendencies = {}
    for name in time_derivative_names:
        tendency = ds[name].diff(dim) / timestep_seconds
        tendency.attrs["units"] = f"{ds[name].units}/s"
        tendency.attrs["long_name"] = f"time derivative of {ds[name].long_name}"
        tendencies[f"tendency_of_{name}"] = tendency
    # drop the first time step since it has no time derivative
    return ds.isel({dim: slice(1, None)}).assign(tendencies)


def compute_column_advective_moisture_tendency(
    ds: xr.Dataset,
    pwat_tendency=f"tendency_of_{TOTAL_WATER_PATH}",
    latent_heat_flux=LATENT_HEAT_FLUX,
    precip=PRECIP_RATE,
    latent_heat_vaporiation=LATENT_HEAT_OF_VAPORIZATION,
) -> xr.Dataset:
    evaporation = ds[latent_heat_flux] / latent_heat_vaporiation
    advective_tendency = ds[pwat_tendency] - evaporation + ds[precip]
    long_name = "tendency of total water path due to advection"
    advective_tendency.attrs["long_name"] = long_name
    return ds.assign({f"{pwat_tendency}_due_to_advection": advective_tendency})


def compute_column_moisture_integral(
    ds: xr.Dataset,
    input_name: str = SPECIFIC_TOTAL_WATER,
    output_name: str = TOTAL_WATER_PATH,
    pressure_thickness_name: str = PRESSURE_THICKNESS,
    dim: str = VERTICAL_DIM,
):
    """Compute total water path."""
    column_integral = (ds[input_name] * ds[pressure_thickness_name]).sum(dim) / GRAVITY
    column_integral.attrs["units"] = "kg/m^2"
    column_integral.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: column_integral})


def assert_column_integral_of_moisture_is_conserved(ds):
    """Assert that the column integral of 'specific_total_water' is close to the
    precipitable water path that is computed online on native grid by FV3GFS. Note
    that there are pretty large difference (rtol=1e-1) which is likely due to the
    fregrid tool doing area-weighted average instead of mass-weighted."""
    expected_pwat = ds[PRECIPITABLE_WATER_PATH]
    integrated_pwat = ds[TOTAL_WATER_PATH]
    print("Mean absolute difference between expected and integrated pwat [kg/m^2]:")
    print(np.abs(expected_pwat - integrated_pwat).mean().values)
    xr.testing.assert_allclose(integrated_pwat, expected_pwat, rtol=1e-1, atol=1e-3)


def assert_global_dry_air_mass_conservation(
    ds,
    dims=HORIZONTAL_DIMS,
    surface_pressure_name=SURFACE_PRESSURE,
    total_water_path_name=TOTAL_WATER_PATH,
    latitude_dim=LATITUDE_DIM,
):
    """Assert that the tendency of global average surface pressure (due to dry air
    only) is close to zero. I.e. dry air mass is conserved."""
    column_dry_air_mass = (
        ds[surface_pressure_name] - ds[total_water_path_name] * GRAVITY
    )
    weights = np.cos(np.deg2rad(ds[latitude_dim]))
    global_dry_air_mass = column_dry_air_mass.weighted(weights).mean(dim=dims)
    global_dry_air_mass_tendency = global_dry_air_mass.diff(TIME_DIM)
    print("Mean absolute global dry air pressure tendency [Pa]:")
    print(np.abs(global_dry_air_mass_tendency).mean().values)
    xr.testing.assert_allclose(
        global_dry_air_mass_tendency,
        xr.zeros_like(global_dry_air_mass_tendency),
        atol=1e-3,
    )


def assert_global_moisture_conservation(
    ds,
    dims=HORIZONTAL_DIMS,
    latitude_dim=LATITUDE_DIM,
    total_water_path_name=TOTAL_WATER_PATH,
    latent_heat_flux_name=LATENT_HEAT_FLUX,
    latent_heat_of_vaporization=LATENT_HEAT_OF_VAPORIZATION,
    precip_rate_name=PRECIP_RATE,
):
    """Assert that the tendency of global average column integrated moisture is equal
    to the global average flux of moisture through the surface."""
    integrated_pwat = ds[total_water_path_name]
    weights = np.cos(np.deg2rad(ds[latitude_dim]))
    global_moisture = integrated_pwat.weighted(weights).mean(dim=dims)
    timestep_seconds = ds[TIME_DIM].diff(TIME_DIM) / np.timedelta64(1, "s")
    actual_global_moisture_tendency = global_moisture.diff(TIME_DIM) / timestep_seconds
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


def construct_lazy_dataset() -> xr.Dataset:
    datasets_dict = open_datasets(DATASET_URLS)
    ds = merge_inputs(INPUT_VARIABLE_NAMES, datasets_dict)
    for var in ds:
        del ds[var].encoding["chunks"]
        del ds[var].encoding["preferred_chunks"]
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    ds = compute_specific_total_water(ds, WATER_SPECIES_NAMES)
    ds = compute_vertical_coarsening(
        ds,
        VERTICALLY_RESOLVED_NAMES,
        VERTICAL_LEVEL_INTERFACES,
    )
    ds = compute_column_moisture_integral(ds)
    ds = compute_tendencies(ds, TIME_DERIVATIVE_NAMES)
    ds = compute_column_advective_moisture_tendency(ds)
    ak_bk_ds = get_coarse_ak_bk(VERTICAL_COORDINATE_URL, VERTICAL_LEVEL_INTERFACES)
    ds = xr.merge([ds, ak_bk_ds])
    ds = ds.chunk(CHUNKS)
    ds.attrs["history"] = (
        "Dataset computed by full-model/projects/fv3gfs_data_process"
        "/compute_vertically_coarsened_data_fv3gfs.py"
        f" script, using following input zarrs: {DATASET_URLS}."
    )
    ds.attrs["vertical_coordinate"] = (
        "The pressure at level interfaces can by computed as "
        "p_i = ak_i + bk_i * PRESsfc, where PRESsfc is the surface pressure and the "
        "p_i pressure corresponds to the interface at the top of the i'th finite "
        "volume layer, counting down from the top of atmosphere."
    )
    return ds


@click.command()
@click.option("--debug", is_flag=True, help="Print metadata instead of writing output.")
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
@click.option("--check-conservation", is_flag=True, help="Check conservation.")
@click.option("-o", "--output", default=OUTPUT_URL, help="URL to write output to.")
@click.option("--n-split", default=65, help="Number of steps to split job over.")
def main(debug, subsample, check_conservation, output, n_split):
    xr.set_options(keep_attrs=True)
    ds = construct_lazy_dataset()
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    if check_conservation:
        assert_column_integral_of_moisture_is_conserved(ds)
        assert_global_dry_air_mass_conservation(ds)
        assert_global_moisture_conservation(ds)
    ds = ds.drop(DROP_VARIABLE_NAMES)
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")
    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        ds.partition.initialize_store(output)
        for i in range(n_split):
            print(f"Writing segment {i + 1} / {n_split}")
            with ProgressBar():
                ds.partition.write(
                    output, n_split, ["time"], i, collect_variable_writes=True
                )


if __name__ == "__main__":
    main()
