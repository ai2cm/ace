# This script is used to compute a training dataset from the "raw"
# E3SMv2 data stored in netCDF form on LC lustre.

# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

# On the Perlmutter cpu partition using --n-workers=32, this script takes about
# 1hr 10min to complete, using a max of about 100 GB of memory for 21 years of
# input data.

import os
import time
import click
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from typing import Callable, List, MutableMapping, Tuple
from glob import glob
from itertools import chain
import numpy as np
import xarray as xr
import xpartition  # noqa: 401


from compute_dataset_fv3gfs import (
    assert_global_dry_air_mass_conservation,
    assert_global_moisture_conservation,
    assert_column_integral_of_moisture_is_conserved,
    compute_specific_total_water,
    compute_vertical_coarsening,
    compute_tendencies,
    compute_column_advective_moisture_tendency,
    compute_column_moisture_integral,
)


INSTANT = "6hourly_instant/1yr"
MEAN = "6hourly/1yr"

# default paths for input/output; can be changed when calling this script
INPUT_DIR = "/global/cfs/cdirs/e3sm/golaz/E3SM/fme/20230614.v2.LR.F2010/post/atm/180x360_gaussian/ts"  # noqa: 501
OUTPUT_URL = "/pscratch/sd/j/jpduncan/ai2/zarr/e3smv2-1deg-gaussian-20yr-fme.zarr"

REFERENCE_PRESSURE = 1e5  # Pa
LIQUID_PRECIP_DENSITY = 1e3  # kg/m**3
LATENT_HEAT_OF_VAPORIZATION = 2.501e6  # J/kg

# 6-hourly instant
SURFACE_PRESSURE = "PS"
SURFACE_TEMPERATURE = "TS"
AIR_TEMPERATURE = "T"
EASTWARD_WIND = "U"
NORTHWARD_WIND = "V"
MEAN_SEA_LEVEL_PRESSURE = "PSL"
GEOPOTENTIAL = "Z"
RELATIVE_HUMIDITY = "RH"
TOTAL_COLUMN_WATER_VAPOR = "TMQ"

# 6-hourly mean
TOTAL_PRECIP_RATE = "PRECT"  # m/s
LATENT_HEAT_FLUX = "LHFLX"

# derived variable names
SPECIFIC_TOTAL_WATER = "specific_total_water"
PRECIPITABLE_WATER_PATH = "precipitable_water_path"  # total from E3SMv2 model outputs
TOTAL_WATER_PATH = "total_water_path"  # computed by vertical integration of 3D vars
PRECIP_RATE = "surface_precipitation_rate"
SURFACE_UP_LONGWAVE_FLUX = "surface_upward_longwave_flux"
SURFACE_UP_SHORTWAVE_FLUX = "surface_upward_shortwave_flux"
TOA_UP_SHORTWAVE_FLUX = "top_of_atmos_upward_shortwave_flux"
PRESSURE_THICKNESS = "pressure_thickness_of_atmospheric_layer"

# dims
TIME_DIM = "time"
HORIZONTAL_DIMS = ["lat", "lon"]
LATITUDE_DIM = "lat"
VERTICAL_DIM = "lev"
VERTICAL_INTERFACE_DIM = "ilev"
HYBRID_LEVEL_COEFFS = ["hyai", "hybi"]

CHUNKS = {"time": 10, "lat": 180, "lon": 360}

# assumed to be found in INSTANT dir
FOURCASTNET_VANILLA = {
    EASTWARD_WIND: ["1000", "850", "500"],
    NORTHWARD_WIND: ["1000", "850", "500"],
    AIR_TEMPERATURE: ["850", "500"],
    GEOPOTENTIAL: ["1000", "500", "850", "050"],
    RELATIVE_HUMIDITY: ["850", "500"],
    SURFACE_PRESSURE: [""],
    "TREFHT": [""],  # temp at 2m
    MEAN_SEA_LEVEL_PRESSURE: [""],
    TOTAL_COLUMN_WATER_VAPOR: [""],
}

# the variables / filename prefixes we need from the raw E3SMv2 output
INPUT_VARIABLE_NAMES = {
    INSTANT: [
        SURFACE_PRESSURE,
        SURFACE_TEMPERATURE,
        AIR_TEMPERATURE,
        EASTWARD_WIND,
        NORTHWARD_WIND,
        "Q",
        "CLDLIQ",
        "CLDICE",
        "RAINQM",
        "SNOWQM",
        TOTAL_COLUMN_WATER_VAPOR,
        "TGCLDLWP",
        "TGCLDIWP",
        "OCNFRAC",
    ],
    MEAN: [
        TOTAL_PRECIP_RATE,
        LATENT_HEAT_FLUX,
        "SHFLX",
        "FLNS",
        "FLDS",
        "FSNS",
        "FSDS",
        "FSNTOA",
        "SOLIN",
        "FLUT",
        # only for water budget dataset:
        "PRECSC",
        "PRECSL",
        "QFLX",
    ],
}

WATER_SPECIES_NAMES = [
    "Q",
    "CLDLIQ",
    "CLDICE",
    "RAINQM",
    "SNOWQM",
]

VARNAMES_3D = [
    AIR_TEMPERATURE,
    EASTWARD_WIND,
    NORTHWARD_WIND,
] + WATER_SPECIES_NAMES

PRECIPITABLE_WATER_PATH_NAMES = [TOTAL_COLUMN_WATER_VAPOR, "TGCLDLWP", "TGCLDIWP"]

VERTICALLY_RESOLVED_NAMES = [
    SPECIFIC_TOTAL_WATER,
    AIR_TEMPERATURE,
    NORTHWARD_WIND,
    EASTWARD_WIND,
]

TIME_DERIVATIVE_NAMES = [TOTAL_WATER_PATH]

# computed here: https://github.com/ai2cm/explore/blob/master/jamesd/2023-06-09-e3smv2-vertical-interface-indices.ipynb  # noqa: 501
VERTICAL_LEVEL_INTERFACES = [
    (0, 19),
    (19, 30),
    (30, 38),
    (38, 44),
    (44, 48),
    (48, 53),
    (53, 61),
    (61, 72),
]

RAD_FLUX_FORMULAS = {
    SURFACE_UP_LONGWAVE_FLUX: (lambda x, y: x + y, "FLNS", "FLDS"),
    SURFACE_UP_SHORTWAVE_FLUX: (lambda x, y: x - y, "FSDS", "FSNS"),
    TOA_UP_SHORTWAVE_FLUX: (lambda x, y: x - y, "SOLIN", "FSNTOA"),
}

DROP_VARIABLE_NAMES = {
    "2D": [  # variables to drop when opening 2D vars
        "time_bnds",
        "lat_bnds",
        "lon_bnds",
        "area",
        "gw",
    ],
    "3D": [  # variables to drop when opening 3D vars
        "P0",
        "PS",
        "time_bnds",
        "lat_bnds",
        "lon_bnds",
        "area",
        "gw",
        "hyam",
        "hybm",
    ],
    "POST": [  # variables to drop at the end
        AIR_TEMPERATURE,
        EASTWARD_WIND,
        NORTHWARD_WIND,
        SPECIFIC_TOTAL_WATER,
        PRESSURE_THICKNESS,
        TOTAL_PRECIP_RATE,
        PRECIPITABLE_WATER_PATH,
        TOTAL_COLUMN_WATER_VAPOR,
        VERTICAL_DIM,
        VERTICAL_INTERFACE_DIM,
        HYBRID_LEVEL_COEFFS[0],
        HYBRID_LEVEL_COEFFS[1],
        "Q",
        "CLDLIQ",
        "CLDICE",
        "RAINQM",
        "SNOWQM",
        "TGCLDLWP",
        "TGCLDIWP",
        "FLNS",
        "FSNS",
        "FSNTOA",
        "PRECSC",
        "PRECSL",
        "QFLX",
    ],
}

# dataset of 2D vars for checking water conservation
WATER_BUDGET_DATASET_VARS = PRECIPITABLE_WATER_PATH_NAMES + [
    SURFACE_PRESSURE,
    PRECIPITABLE_WATER_PATH,
    TOTAL_PRECIP_RATE,
    LATENT_HEAT_FLUX,
    "PRECSC",
    "PRECSL",
    "QFLX",
]


def expand_names_by_level(variables: MutableMapping[str, List[str]]) -> List[str]:
    names = []
    for var_name, levels in variables.items():
        names.extend([f"{var_name}{lev}" for lev in levels])
    return names


def get_nc_paths(base_dir: str, var_names: List[str]) -> MutableMapping[str, List[str]]:
    paths = {
        var_name: sorted(list(glob(os.path.join(base_dir, f"{var_name}_*.nc"))))
        for var_name in var_names
    }
    return paths


def open_dataset(
    dataset_dirs: MutableMapping[str, str],
    input_variable_names: MutableMapping[str, List[str]] = INPUT_VARIABLE_NAMES,
    varnames_3d: List[str] = VARNAMES_3D,
    drop_variable_names: MutableMapping[str, List[str]] = DROP_VARIABLE_NAMES,
    chunks: MutableMapping[str, int] = CHUNKS,
    vanilla: bool = False,
) -> xr.Dataset:
    """Open datasets from NetCDF files in directory that match input variable names."""
    if vanilla:
        var_names = expand_names_by_level(FOURCASTNET_VANILLA)
        var_paths = get_nc_paths(dataset_dirs[INSTANT], var_names)
    else:
        var_paths = {}
        for key in dataset_dirs.keys():
            var_paths.update(get_nc_paths(dataset_dirs[key], input_variable_names[key]))
    print(
        f"Opening {len(list(chain.from_iterable(var_paths.values())))} files with "
        f"{len(var_paths.keys())} vars..."
    )
    datasets = {}
    start = time.time()
    for varname, paths in var_paths.items():
        var_start = time.time()
        if varname in varnames_3d:
            drop_vars = drop_variable_names["3D"]
        else:
            drop_vars = drop_variable_names["2D"]
        datasets[varname] = xr.open_mfdataset(
            paths,
            chunks=chunks,
            data_vars="minimal",
            coords="minimal",
            parallel=True,
        ).drop(drop_vars, errors="ignore")
        print(f"{varname} files opened in {time.time() - var_start:.2f} s...")
    print(f"All files opened in {time.time() - start:.2f} s. Merging...")
    return xr.merge(datasets.values(), compat="override", join="override")


def compute_pressure_thickness(
    ds: xr.Dataset,
    vertical_dim: str = VERTICAL_DIM,
    vertical_interface_dim: str = VERTICAL_INTERFACE_DIM,
    hybrid_level_coeffs: List[str] = HYBRID_LEVEL_COEFFS,
    reference_pressure: float = REFERENCE_PRESSURE,
    surface_pressure: str = SURFACE_PRESSURE,
    output_name: str = PRESSURE_THICKNESS,
):
    hyai, hybi = hybrid_level_coeffs
    sfc_pressure = ds[surface_pressure].expand_dims(
        {vertical_interface_dim: ds[vertical_interface_dim]}, axis=3
    )
    phalf = sfc_pressure * ds[hybi] + reference_pressure * ds[hyai]
    thickness = (
        phalf.diff(dim=vertical_interface_dim)
        .rename({vertical_interface_dim: vertical_dim})
        .rename(output_name)
        .assign_coords({vertical_dim: (vertical_dim, ds[vertical_dim].values)})
    )
    thickness.attrs["units"] = "Pa"
    thickness.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: thickness})


def compute_surface_precipitation_rate(
    ds: xr.Dataset,
    total_precip_rate_name: str = TOTAL_PRECIP_RATE,
    liquid_precip_density: float = LIQUID_PRECIP_DENSITY,
    output_name: str = PRECIP_RATE,
):
    precip_mass_flux = ds[total_precip_rate_name] * liquid_precip_density
    precip_mass_flux.attrs["units"] = "kg/m2/s"
    precip_mass_flux.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: precip_mass_flux})


def compute_precipitable_water_path(
    ds: xr.Dataset,
    precipitable_water_path_names: List[str] = PRECIPITABLE_WATER_PATH_NAMES,
    output_name: str = PRECIPITABLE_WATER_PATH,
):
    water_path = sum([ds[name] for name in precipitable_water_path_names])
    water_path.attrs["units"] = "kg/m2"
    water_path.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: water_path})


def compute_rad_fluxes(
    ds: xr.Dataset,
    rad_flux_formulas: MutableMapping[
        str, Tuple[Callable, str, str]
    ] = RAD_FLUX_FORMULAS,
) -> xr.Dataset:
    fluxes = {}
    for output_name, formula in rad_flux_formulas.items():
        fluxes[output_name] = formula[0](ds[formula[1]], ds[formula[2]])
        fluxes[output_name].attrs["long_name"] = output_name.replace("_", " ")
        fluxes[output_name].attrs["units"] = ds[formula[1]].attrs["units"]
    return ds.assign(fluxes)


def construct_lazy_dataset(
    dataset_dirs: MutableMapping[str, str], vanilla: bool = False
) -> xr.Dataset:
    start = time.time()
    print(f"Opening dataset...")
    ds = open_dataset(dataset_dirs, vanilla=vanilla)
    print(f"Dataset opened in {time.time() - start:.2f} s total.")
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    if not vanilla:
        ds = compute_pressure_thickness(ds)
        ds = compute_rad_fluxes(ds)
        ds = compute_surface_precipitation_rate(ds)
        ds = compute_precipitable_water_path(ds)  # only used for conservation check
        ds = compute_specific_total_water(ds, WATER_SPECIES_NAMES, SPECIFIC_TOTAL_WATER)
        ds = compute_vertical_coarsening(
            ds,
            VERTICALLY_RESOLVED_NAMES,
            VERTICAL_LEVEL_INTERFACES,
            VERTICAL_DIM,
            PRESSURE_THICKNESS,
        )
        ds = compute_column_moisture_integral(
            ds,
            SPECIFIC_TOTAL_WATER,
            TOTAL_WATER_PATH,
            PRESSURE_THICKNESS,
            VERTICAL_DIM,
        )
        ds[TOTAL_WATER_PATH].attrs["units"] = "kg/m2"  # change to E3SMv2 format
        ds = compute_tendencies(ds, TIME_DERIVATIVE_NAMES, TIME_DIM)
        ds = compute_column_advective_moisture_tendency(
            ds,
            f"tendency_of_{TOTAL_WATER_PATH}",
            LATENT_HEAT_FLUX,
            PRECIP_RATE,
            LATENT_HEAT_OF_VAPORIZATION,
        )
        ds_dirs = list(dataset_dirs.values())
    else:
        ds_dirs = [dataset_dirs[INSTANT]]
    ds = ds.chunk(CHUNKS).astype(np.float32)
    ds.attrs["history"] = (
        "Dataset computed by full-model/projects/e3smv2_data_process/compute_dataset_e3smv2.py"  # noqa: 501
        f" script, inputs from the following directories : {ds_dirs}."
    )
    return ds


@click.command()
@click.option("--debug", is_flag=True, help="Print metadata instead of writing output.")
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
@click.option("--vanilla", is_flag=True, help="Compute vanilla FourCastNet dataset.")
@click.option("--check-conservation", is_flag=True, help="Check conservation.")
@click.option(
    "--water-budget-dataset",
    is_flag=True,
    help="Create a dataset of 2D vars for checking the water budget.",
)
@click.option(
    "-i", "--input-dir", default=INPUT_DIR, help="Directory in which to find input ncs."
)
@click.option("-o", "--output", default=OUTPUT_URL, help="URL to write output to.")
@click.option("--n-split", default=100, help="Number of steps to split job over.")
@click.option("--n-workers", default=4, help="Number of Dask workers.")
def main(
    debug,
    subsample,
    vanilla,
    check_conservation,
    water_budget_dataset,
    input_dir,
    output,
    n_split,
    n_workers,
):
    xr.set_options(keep_attrs=True)
    _ = Client(n_workers=n_workers)

    dataset_dirs = {
        INSTANT: os.path.join(input_dir, INSTANT),
        MEAN: os.path.join(input_dir, MEAN),
    }
    ds = construct_lazy_dataset(dataset_dirs, vanilla)
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    if check_conservation:
        # these currently fail
        assert_column_integral_of_moisture_is_conserved(ds)
        assert_global_dry_air_mass_conservation(
            ds,
            dims=HORIZONTAL_DIMS,
            surface_pressure_name=SURFACE_PRESSURE,
            total_water_path_name=TOTAL_WATER_PATH,
            latitude_dim=LATITUDE_DIM,
        )
        assert_global_moisture_conservation(
            ds,
            dims=HORIZONTAL_DIMS,
            latitude_dim=LATITUDE_DIM,
            total_water_path_name=TOTAL_WATER_PATH,
            latent_heat_flux_name=LATENT_HEAT_FLUX,
            latent_heat_of_vaporization=LATENT_HEAT_OF_VAPORIZATION,
            precip_rate_name=PRECIP_RATE,
        )
    if water_budget_dataset:
        ds = ds[WATER_BUDGET_DATASET_VARS]
    else:
        ds = ds.drop(DROP_VARIABLE_NAMES["POST"], errors="ignore")
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
