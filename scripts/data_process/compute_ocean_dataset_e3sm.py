import time

import cftime
import click
import numpy as np
import xarray as xr
from compute_ocean_dataset import OceanDatasetComputationConfig, OceanDatasetConfig
from dask.diagnostics import ProgressBar
from ocean_emulators.preprocessing import spatially_filter
from xgcm import Grid

LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg
_LATENT_HEAT_VAPORIZATION_0_C = 2.5e6  # J/kg
_SPECIFIC_ENTHALPY_LIQUID = 4185.5
_SPECIFIC_ENTHALPY_VAP0R = 1846
_FREEZING_TEMPERATURE = 273.15


def latent_heat_vaporization(T):
    return _LATENT_HEAT_VAPORIZATION_0_C + (
        _SPECIFIC_ENTHALPY_LIQUID - _SPECIFIC_ENTHALPY_VAP0R
    ) * (T - _FREEZING_TEMPERATURE)


def open_ocean_dataset(config):
    ds = xr.open_mfdataset(config.ocean_dataset_nc_files)
    ds_monthly_thickness = xr.open_mfdataset(
        config.ocean_dataset_monthly_layer_thickness_files
    )
    upsampled = ds_monthly_thickness.interp(
        Time=ds.Time.values, method="nearest", kwargs={"fill_value": "extrapolate"}
    )
    ds["layer_thickness"] = upsampled[config.standard_names.ocean_layer_thickness]
    mesh = xr.open_dataset(config.nc_grid_path)
    ds["refBottomDepth"] = mesh.refBottomDepth
    wetmask = np.isnan(mesh.temperature.isel(Time=0))
    ds["wetmask"] = wetmask
    ds.wetmask.attrs["long_name"] = "wet mask"
    ds.wetmask.attrs["units"] = ""
    return ds


def open_ice_dataset(config):
    ds = xr.open_mfdataset(
        config.ice_dataset_nc_files,
        concat_dim=config.standard_names.time_dim,
        combine="nested",
    )
    return ds


def open_dataset(config):
    ocean = open_ocean_dataset(config)
    ice = open_ice_dataset(config)
    ds = xr.merge([ocean, ice])
    return ds


def get_no_spatially_filtered_variables(ds):
    return ds[["refBottomDepth", "wetmask", "layer_thickness"]]


def compute_surface_downward_heat_flux(ds, ocean_stream_prefix="timeCustom_avg_"):
    """In CM4, hfds is defined as
    Surface ocean heat flux from
    SW+LW+latent+sensible+masstransfer+frazil+seaice_melt_heat.
    In E3SM, we compute this as the sum of the following variables:
    shortWaveHeatFlux, longWaveHeatFluxDown
    evaporationFlux, sensibleHeatFlux,
    """
    ocean_downward_heat_flux = (
        ds[ocean_stream_prefix + "shortWaveHeatFlux"]
        + ds[ocean_stream_prefix + "longWaveHeatFluxDown"]
        + ds[ocean_stream_prefix + "evaporationFlux"] * LATENT_HEAT_OF_VAPORIZATION
        + ds[ocean_stream_prefix + "sensibleHeatFlux"]
    )
    return ocean_downward_heat_flux


def compute_vertical_coarsening(ds: xr.Dataset, config, variable_names) -> xr.Dataset:
    standard_names = config.standard_names
    target_vertical_levels = np.array(config.ocean_vertical_target_interface_levels)
    var2D = []
    var3D = []
    for var in variable_names:
        if ds[var].dims == (
            standard_names.time_dim,
            standard_names.latitude_dim,
            standard_names.longitude_dim,
        ):
            var2D.append(var)
        elif ds[var].dims == (
            standard_names.time_dim,
            standard_names.vertical_dim,
            standard_names.latitude_dim,
            standard_names.longitude_dim,
        ):
            var3D.append(var)
        else:
            print(f"Skipping {var} since it is a coordinate variable.")
    ds_2d = ds[var2D].copy()
    lev = ds.refBottomDepth
    lev_outer = np.zeros(len(lev))
    lev_outer[0] = lev.values[0] * 0.5
    lev_outer[1:] = 0.5 * (lev.values[0:-1] + lev.values[1:])
    lev_outer = np.append(lev_outer, lev.values[-1] * 2.0 - lev_outer[-1])
    ds = ds.rename_dims({config.standard_names.vertical_dim: "lev"})
    ds = ds.assign_coords({"lev": lev, "lev_outer": lev_outer})
    grid = Grid(
        ds,
        coords={"Z": {"center": "lev", "outer": "lev_outer"}},
        boundary="fill",
        periodic=False,
        autoparse_metadata=False,
    )
    dz = ds["layer_thickness"]
    ds_extensive = ds[var3D] * dz
    ds_extensive = ds_extensive.assign_coords({"lev_outer": lev_outer})
    ds_extensive_regridded = xr.Dataset()
    for var in var3D:
        ds_extensive_regridded[var] = grid.transform(
            ds_extensive[var],
            "Z",
            target_vertical_levels,
            target_data=ds.lev_outer,
            method="conservative",
        )
    ds_extensive_regridded = ds_extensive_regridded.rename({"lev_outer": "lev"})
    dz_regridded = xr.DataArray(
        np.diff(target_vertical_levels),
        dims=["lev"],
        coords={"lev": ds_extensive_regridded.lev},
    )
    ds_regridded = ds_extensive_regridded / dz_regridded
    ds_regridded = ds_regridded.assign_coords(dz=dz_regridded)
    for co_name, co in ds.coords.items():
        if "lev" not in co.dims:
            ds_regridded = ds_regridded.assign_coords({co_name: co})
    ds_regridded = ds_regridded.drop_vars("lev_outer")
    ds_regridded = ds_regridded.drop_vars("lev").rename_dims(
        {"lev": config.standard_names.vertical_dim}
    )
    ds_regridded = ds_regridded.drop_vars("dz")
    ds_regridded = xr.merge([ds_regridded, ds_2d])
    return ds_regridded


def rename_and_split_dataset_to_individual_layer(ds, config):
    rename = config.renaming
    ds = ds.rename(rename)
    for var in ds.variables:
        if config.standard_names.vertical_dim in ds[var].dims:
            for i in range(len(ds[config.standard_names.vertical_dim])):
                ds[f"{var}_{i}"] = ds[var].isel({config.standard_names.vertical_dim: i})
            ds = ds.drop_vars(var)
    return ds


def get_vertical_coarsened_wetmask(ds, config):
    standard_names = config.standard_names
    for var in ds.variables:
        if standard_names.sea_water_potential_temperature in var:
            wetmask = np.isnan(ds[var].isel(time=0))
            current_level = var.split("_")[-1]
            ds["wetmask_" + current_level] = wetmask

    return ds


def reindex_time_with_xtime(ds, config):
    timestamps = []
    for i in range(len(ds.xtime)):
        date_string = ds.xtime.values[i].decode("utf-8")
        date_part, time_part = date_string.split("_")
        year, month, day = date_part.split("-")
        hour, minute, _ = time_part.split(":")
        timestamps.append(
            cftime.DatetimeNoLeap(
                int(year), int(month), int(day), int(hour), int(minute)
            )
        )
        ds = ds.assign_coords({config.standard_names.time_dim: timestamps})
        return ds


def construct_lazy_dataset(
    config: OceanDatasetComputationConfig,
) -> xr.Dataset:
    start = time.time()
    standard_names = config.standard_names
    print(f"Opening dataset...")
    ds = open_dataset(config)
    print(f"Dataset opened in {time.time() - start:.2f} s total.")
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    ds = reindex_time_with_xtime(ds, config)
    spatially_filtered_variables = list(config.renaming.keys())
    if config.compute_e3sm_surface_downward_heat_flux:
        ds["sfc_heat_flux"] = compute_surface_downward_heat_flux(ds)
        spatially_filtered_variables.append("sfc_heat_flux")
    ds_unfiltered = get_no_spatially_filtered_variables(ds)
    ds_filtered = spatially_filter(
        ds[spatially_filtered_variables],
        ds["wetmask"],
        depth_dim=standard_names.vertical_dim,
        y_dim=standard_names.latitude_dim,
        x_dim=standard_names.longitude_dim,
    )
    ds = xr.merge([ds_unfiltered, ds_filtered])
    ds = compute_vertical_coarsening(ds, config, spatially_filtered_variables)
    ds = rename_and_split_dataset_to_individual_layer(ds, config)
    ds = get_vertical_coarsened_wetmask(ds, config)
    renamed_standard_names = config.standard_names
    renamed_standard_names.time_dim = "time"
    chunks = config.chunking.get_chunks(renamed_standard_names)
    ds = ds.chunk(chunks).astype(np.float32)
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts"
        "/compute_ocean_dataset_e3smv2.py"
        f" script, using inputs from the following: {config.ocean_dataset_nc_files}"
        f" and {config.ice_dataset_nc_files}."
    )
    return ds


@click.command()
@click.option("--config", help="Path to dataset configuration YAML file.")
@click.option("-o", "--output", help="URL to write output to.")
@click.option("--debug", is_flag=True, help="Print metadata instead of writing output.")
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
@click.option("--n-workers", default=4, help="Number of Dask workers.")
@click.option("--ranks", default=16, help="total number of available ranks")
@click.option("--rank", default=0, help="rank of job")
def main(
    config,
    output,
    debug,
    subsample,
    n_workers,
    ranks,
    rank,
):
    xr.set_options(keep_attrs=True)
    # _ = Client(n_workers=n_workers, threads_per_worker=2)
    config = OceanDatasetConfig.from_file(config).dataset_computation
    ds = construct_lazy_dataset(config)
    if subsample:
        ds = ds.isel(time=slice(10, 13))
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
