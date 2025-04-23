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
from glob import glob
from itertools import chain
from typing import Callable, List, MutableMapping, Optional, Sequence, Tuple

import click
import numpy as np
import xarray as xr
import xpartition  # noqa: F401
from compute_dataset import (
    DatasetComputationConfig,
    DatasetConfig,
    assert_column_integral_of_moisture_is_conserved,
    assert_global_dry_air_mass_conservation,
    assert_global_moisture_conservation,
    compute_column_advective_moisture_tendency,
    compute_column_moisture_integral,
    compute_specific_total_water,
    compute_tendencies,
    compute_vertical_coarsening,
)
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from xtorch_harmonics import roundtrip_filter

# default paths for input/output; can be changed when calling this script
INPUT_DIR = "/global/cfs/cdirs/e3sm/golaz/E3SM/fme/20230614.v2.LR.F2010/post/atm/180x360_gaussian/ts"  # noqa: E501
OUTPUT_URL = "/pscratch/sd/j/jpduncan/ai2/zarr/2023-11-22-e3smv2-vertically-resolved-1deg-fme-dataset.zarr"  # noqa: E501

REFERENCE_PRESSURE = 1e5  # Pa
LIQUID_PRECIP_DENSITY = 1e3  # kg/m**3
LATENT_HEAT_OF_VAPORIZATION = 2.501e6  # J/kg

# derived variable names
SURFACE_UP_LONGWAVE_FLUX = "surface_upward_longwave_flux"
SURFACE_UP_SHORTWAVE_FLUX = "surface_upward_shortwave_flux"
TOA_UP_SHORTWAVE_FLUX = "top_of_atmos_upward_shortwave_flux"

RAD_FLUX_FORMULAS = {
    SURFACE_UP_LONGWAVE_FLUX: (lambda x, y: x + y, "FLNS", "FLDS"),
    SURFACE_UP_SHORTWAVE_FLUX: (lambda x, y: x - y, "FSDS", "FSNS"),
    TOA_UP_SHORTWAVE_FLUX: (lambda x, y: x - y, "SOLIN", "FSNTOA"),
}

SURFACE_PRECIPITATION = "surface_precipitation_rate"
PRECIPITABLE_WATER_PATH_NAMES = ["TMQ", "TGCLDLWP", "TGCLDIWP"]

DROP_VARIABLE_NAMES = {
    "2D": [  # variables to drop when opening 2D vars
        "time_bnds",
        "lat_bnds",
        "lon_bnds",
        "area",
        "gw",
        "elevation",
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
}


def expand_names_by_level(variables: MutableMapping[str, List[str]]) -> List[str]:
    names = []
    for var_name, levels in variables.items():
        names.extend([f"{var_name}{lev}" for lev in levels])
    return names


def get_nc_paths(
    base_dir: str, var_names: Sequence[str]
) -> MutableMapping[str, List[str]]:
    paths = {
        var_name: sorted(list(glob(os.path.join(base_dir, f"{var_name}_*.nc"))))
        for var_name in var_names
    }
    return paths


def get_time_invariant_nc_paths(
    base_dir: Optional[str],
) -> MutableMapping[str, List[str]]:
    paths = {
        "time_invariant": list(glob(os.path.join(base_dir, f"*.nc")))  # type: ignore
    }
    return paths


def open_dataset(
    dataset_dirs: MutableMapping[str, str],
    config: DatasetComputationConfig,
) -> xr.Dataset:
    """Open datasets from NetCDF files in directory that match input variable names."""
    var_paths: MutableMapping[str, List[str]] = {}
    input_variable_names = config.variable_sources
    for key in dataset_dirs.keys():
        var_paths.update(get_nc_paths(dataset_dirs[key], input_variable_names[key]))
    var_paths.update(get_time_invariant_nc_paths(config.time_invariant_dir))
    print(
        f"Opening {len(list(chain.from_iterable(var_paths.values())))} files with "
        f"{len(var_paths.keys())} vars..."
    )
    standard_names = config.standard_names
    varnames_3D = [
        standard_names.air_temperature,
        standard_names.eastward_wind,
        standard_names.northward_wind,
    ] + standard_names.water_species
    chunks = config.chunking.get_chunks(config.standard_names)
    datasets = {}
    start = time.time()
    if "time_invariant" in var_paths:
        for path in var_paths["time_invariant"]:
            ds = xr.open_dataset(path, decode_timedelta=False).drop(
                DROP_VARIABLE_NAMES["2D"], errors="ignore"
            )
            if "time" in ds.coords:
                ds = ds.isel(time=0, drop=True)
            for varname in input_variable_names["time_invariant"]:
                if varname not in datasets and varname in ds.variables:
                    datasets[varname] = ds
        del var_paths["time_invariant"]
    for varname, paths in var_paths.items():
        var_start = time.time()
        if varname in varnames_3D:
            drop_vars = DROP_VARIABLE_NAMES["3D"]
        else:
            drop_vars = DROP_VARIABLE_NAMES["2D"]
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
    vertical_dim: str,
    vertical_interface_dim: str,
    hybrid_level_coeffs: List[str],
    reference_pressure: float,
    surface_pressure: str,
    output_name: str,
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


def compute_coarse_ak_bk(
    ds: xr.Dataset,
    interface_indices: Sequence[Tuple[int, int]],
    z_dim: str,
    hybrid_level_coeffs: List[str],
    reference_pressure: float = REFERENCE_PRESSURE,
):
    """Return dataset with scalar ak and bk coordinates that define coarse
    interfaces, which should match the conventions used in compute_dataset_fv3gfs.py.

    Args:
        ds: xr.Dataset with hybrid level coefficients named like hybrid_level_coeffs.
        interface_indices: list of tuples of indices of the interfaces in the vertical.
        z_dim: name of dimension along which ak and bk are defined.
        hybrid_level_coeffs: Hybrid level coeff names in ds, ordered like ["ak", "bk"].

    Returns:
        xr.Dataset with ak and bk variables as scalars labeled as ak_0, bk_0, etc.

    Note:
        The ak and bk variables will have one more vertical level than the other 3D
        variables since they represent the interfaces between levels.

    """
    data = {}
    hyai, hybi = hybrid_level_coeffs
    for i, (start, end) in enumerate(interface_indices):
        data[f"ak_{i}"] = ds[hyai].isel({z_dim: start}) * reference_pressure
        data[f"bk_{i}"] = ds[hybi].isel({z_dim: start})
        if i == len(interface_indices) - 1:
            data[f"ak_{i + 1}"] = ds[hyai].isel({z_dim: end}) * reference_pressure
            data[f"bk_{i + 1}"] = ds[hybi].isel({z_dim: end})
    for i in range(len(interface_indices) + 1):
        data[f"ak_{i}"].attrs["units"] = "Pa"
        data[f"bk_{i}"].attrs["units"] = ""  # unitless quantity
        for name in ["ak", "bk"]:
            data[f"{name}_{i}"] = data[f"{name}_{i}"].drop(z_dim)
    return xr.Dataset(data).compute()


def compute_surface_precipitation_rate(
    ds: xr.Dataset,
    total_precip_rate_name,
    liquid_precip_density: float = LIQUID_PRECIP_DENSITY,
    output_name: str = SURFACE_PRECIPITATION,
):
    precip_mass_flux = ds[total_precip_rate_name] * liquid_precip_density
    precip_mass_flux.attrs["units"] = "kg/m2/s"
    precip_mass_flux.attrs["long_name"] = output_name.replace("_", " ")
    return ds.assign({output_name: precip_mass_flux})


def compute_precipitable_water_path(
    ds: xr.Dataset,
    output_name: str,
    precipitable_water_path_names: List[str],
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
    config: DatasetComputationConfig,
    dataset_dirs: MutableMapping[str, str],
) -> xr.Dataset:
    start = time.time()
    standard_names = config.standard_names
    print(f"Opening dataset...")
    ds = open_dataset(dataset_dirs, config)
    print(f"Dataset opened in {time.time() - start:.2f} s total.")
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    if config.roundtrip_fraction_kept is not None:
        ds = roundtrip_filter(
            ds,
            lat_dim=standard_names.latitude_dim,
            lon_dim=standard_names.longitude_dim,
            fraction_modes_kept=config.roundtrip_fraction_kept,
        )

    ds = compute_pressure_thickness(
        ds,
        vertical_dim=standard_names.vertical_dim,
        vertical_interface_dim=standard_names.vertical_interface_dim,
        hybrid_level_coeffs=standard_names.hybrid_level_coeffs,
        reference_pressure=REFERENCE_PRESSURE,
        surface_pressure=standard_names.surface_pressure,
        output_name=standard_names.pressure_thickness,
    )
    ds = compute_rad_fluxes(ds)
    ds = compute_surface_precipitation_rate(
        ds,
        total_precip_rate_name=standard_names.precip_rate,
    )
    water_species_name = [
        item for item in standard_names.water_species if item.lower() != "none"
    ]
    ds = compute_specific_total_water(
        ds,
        water_condensate_names=water_species_name,
        output_name=standard_names.specific_total_water,
    )
    ds = compute_vertical_coarsening(
        ds,
        vertically_resolved_names=standard_names.vertically_resolved,
        interface_indices=config.vertical_coarsening_indices,
        dim=standard_names.vertical_dim,
        pressure_thickness_name=standard_names.pressure_thickness,
        validate_indices=config.validate_vertical_coarsening_indices,
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
    ak_bk_ds = compute_coarse_ak_bk(
        ds,
        interface_indices=config.vertical_coarsening_indices,
        z_dim=standard_names.vertical_interface_dim,
        hybrid_level_coeffs=standard_names.hybrid_level_coeffs,
    )
    ds = xr.merge([ds, ak_bk_ds])
    ds_dirs = list(dataset_dirs.values())
    chunks = config.chunking.get_chunks(config.standard_names)
    ds = ds.chunk(chunks).astype(np.float32)
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts/e3smv2_data_process"
        "/compute_dataset_e3smv2.py"
        f" script, using inputs from the following directories: {ds_dirs}."
    )
    ds.attrs["vertical_coordinate"] = (
        "The pressure at level interfaces can by computed as "
        "p_i = ak_i + bk_i * PS, where PS is the surface pressure and the "
        "p_i pressure corresponds to the interface at the top of the i'th finite "
        "volume layer, counting down from the top of atmosphere."
    )
    return ds


@click.command()
@click.option("--config", help="Path to dataset configuration YAML file.")
@click.option(
    "-i",
    "--input-dir",
    default=INPUT_DIR,
    help="Directory in which to find time-varying input ncs.",
)
@click.option("-o", "--output", default=OUTPUT_URL, help="URL to write output to.")
@click.option("--debug", is_flag=True, help="Print metadata instead of writing output.")
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
@click.option("--check-conservation", is_flag=True, help="Check conservation.")
@click.option(
    "--water-budget-dataset",
    is_flag=True,
    help="Create a dataset of 2D vars for checking the water budget.",
)
@click.option("--n-workers", default=4, help="Number of Dask workers.")
def main(
    config,
    input_dir,
    output,
    debug,
    subsample,
    check_conservation,
    water_budget_dataset,
    n_workers,
):
    xr.set_options(keep_attrs=True)
    _ = Client(n_workers=n_workers)
    config = DatasetConfig.from_file(config).dataset_computation
    standard_names = config.standard_names
    dataset_dirs = {}
    for key in config.variable_sources.keys():
        if key != "time_invariant":
            dataset_dirs[key] = os.path.join(input_dir, key)
    ds = construct_lazy_dataset(config, dataset_dirs)
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    if check_conservation:
        # these currently fail
        ds = compute_precipitable_water_path(
            ds,
            output_name=standard_names.precipitable_water_path,
            precipitable_water_path_names=PRECIPITABLE_WATER_PATH_NAMES,
        )
        assert_column_integral_of_moisture_is_conserved(
            ds, standard_names.precipitable_water_path, standard_names.total_water_path
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
    if water_budget_dataset:
        water_budget_dataset_vars = [
            standard_names.surface_pressure,
            standard_names.precipitable_water_path,
            standard_names.precip_rate,
            standard_names.latent_heat_flux,
            "PRECSC",
            "PRECSL",
            "QFLX",
            "TMQ",
            "TGCLDLWP",
            "TGCLDIWP",
        ]

        ds = ds[water_budget_dataset_vars]
    else:
        dropped_variables = (
            [
                item
                for item in standard_names.dropped_variables
                if item.lower() != "none"
            ]
            + standard_names.hybrid_level_coeffs
            + [
                standard_names.precip_rate,
                standard_names.vertical_interface_dim,
                "TMQ",
                "TGCLDLWP",
                "TGCLDIWP",
                "FLNS",
                "FSNS",
                "FSNTOA",
                "PRECSC",
                "PRECSL",
                "QFLX",
            ]
        )
        ds = ds.drop(dropped_variables, errors="ignore")
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")
    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        ds.partition.initialize_store(output)
        for i in range(config.n_split):
            print(f"Writing segment {i + 1} / {config.n_split}")
            with ProgressBar():
                ds.partition.write(
                    output, config.n_split, ["time"], i, collect_variable_writes=True
                )


if __name__ == "__main__":
    main()
