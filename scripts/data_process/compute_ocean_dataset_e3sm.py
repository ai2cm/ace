import re
import time

import click
import numpy as np
import xarray as xr
from compute_ocean_dataset import (
    OceanDatasetComputationConfig,
    OceanDatasetConfig,
    clear_compressors_encoding,
)
from dask.diagnostics import ProgressBar
from ocean_emulators.preprocessing import spatially_filter

LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg


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


def rename_and_split_dataset_to_individual_layer(ds, config):
    rename = config.renaming
    ds = ds.rename(rename)
    for var in ds.variables:
        if "depth_center" in ds[var].dims:
            for i in range(len(ds["depth_center"])):
                ds[f"{var}_{i}"] = ds[var].isel({"depth_center": i})
            ds = ds.drop_vars(var)
    return ds


def get_vertical_coarsened_wetmask(ds, config):
    standard_names = config.standard_names
    for var in ds.variables:
        if standard_names.sea_water_potential_temperature in var:
            wetmask = ~ds[var].isnull().any(dim="time")
            current_level = var.split("_")[-1]
            ds["mask_" + current_level] = wetmask
    ds["mask_2d"] = ~ds["sst"].isnull().any(dim="time")
    return ds


def get_mask_for(name: str):
    LEVEL_PATTERN = re.compile(r"_(\d+)$")
    match = LEVEL_PATTERN.search(name)
    if match:
        # 3D variable
        level = int(match.group(1))
        return f"mask_{level}"
    else:
        return "mask_2d"


def ensure_nans_outside_mask(ds, config):
    standard_names = config.standard_names
    for var in ds.data_vars:
        if ds[var].dims == (
            "time",
            standard_names.latitude_dim,
            standard_names.longitude_dim,
        ):
            mask_name = f"mask_{var}"
            if mask_name not in ds:
                mask_name = get_mask_for(var)
            mask = ds[mask_name].broadcast_like(ds[var])
            ds[var] = ds[var].where(mask == 1, np.nan)
    return ds


def add_idepths(ds, config):
    """Add the interface depths to the dataset"""
    idepths = config.ocean_vertical_target_interface_levels
    for i in range(len(idepths)):
        ds[f"idepth_{i}"] = xr.DataArray(idepths[i], dims=None)
    return ds


def restore_original_attrs(ds_original, ds_new):
    for var in ds_original.data_vars:
        if var in ds_new:
            ds_new[var].attrs = ds_original[var].attrs
    return ds_new


def mask_out_sea_surface_height(ds, config, threshold=-4):
    """
    Mask out sea surface height that are above a threshold.
    This is done to avoid points below sea ice where sea surface height has
    large negative values.
    """
    ssh_name = config.renaming[config.standard_names.sea_surface_height]
    ssh_mean = ds[ssh_name].mean(dim="time")
    mask_zos = xr.where((ssh_mean > threshold | np.isnan(ssh_mean)), 1, 0)
    ds[f"mask_{ssh_name}"] = mask_zos
    return ds


def construct_lazy_dataset(
    config: OceanDatasetComputationConfig,
) -> xr.Dataset:
    start = time.time()
    standard_names = config.standard_names
    print(f"Opening dataset...")
    ds = xr.open_mfdataset(config.ocean_dataset_nc_files)
    print(f"Dataset opened in {time.time() - start:.2f} s total.")
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    ds["sfc_heat_flux"] = compute_surface_downward_heat_flux(ds)
    if config.spatial_filter.enabled:
        spatially_filtered_variables = list(config.renaming.keys())
        spatially_filtered_variables.append("sfc_heat_flux")
        spatially_filtered_variables.append("sst")
        wetmask = np.isnan(ds[standard_names.wetmask_reference_variable].isel(time=0))
        ds[standard_names.wetmask] = wetmask
        ds[standard_names.wetmask].attrs["long_name"] = "wet mask"
        ds[standard_names.wetmask].attrs["units"] = ""
        unfiltered_variables = [
            standard_names.wetmask
        ] + config.spatial_filter.exclude_from_filtering
        ds_unfiltered = ds[unfiltered_variables]
        for var in config.spatial_filter.exclude_from_filtering:
            spatially_filtered_variables.remove(var)
        ds_filtered = spatially_filter(
            ds[spatially_filtered_variables],
            ds[standard_names.wetmask],
            filter_scale=config.spatial_filter.filter_scale,
            depth_dim=standard_names.vertical_dim,
            y_dim=standard_names.latitude_dim,
            x_dim=standard_names.longitude_dim,
        )
        ds_filtered = restore_original_attrs(ds, ds_filtered)
        ds = xr.merge([ds_unfiltered, ds_filtered])
        ds = ds.drop_vars([standard_names.wetmask])
    ds = ds.drop_vars([config.standard_names.ocean_layer_thickness])
    sharding = config.sharding.get_chunks(standard_names)
    ds = ds.chunk(sharding)
    ds = rename_and_split_dataset_to_individual_layer(ds, config)
    ds = mask_out_sea_surface_height(ds, config)
    ds = get_vertical_coarsened_wetmask(ds, config)
    ds = ensure_nans_outside_mask(ds, config)
    ds = add_idepths(ds, config)
    ds = config.shift_timestamps(ds)
    ds = ds.astype(np.float32)
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts/data_process"
        "/compute_ocean_dataset_e3sm.py"
        f" script, using inputs from the following: {config.ocean_dataset_nc_files}."
    )
    return ds


@click.command()
@click.option("--config", help="Path to dataset configuration YAML file.")
@click.option("--output-store", help="Path to output zarr store.")
@click.option(
    "--debug",
    is_flag=True,
    help="Print metadata and return QC plots instead of writing output.",
)
@click.option("--subsample", is_flag=True, help="Subsample the data before writing.")
def main(
    config,
    output_store,
    debug,
    subsample,
):
    print(f"--output-store is {output_store}")
    xr.set_options(keep_attrs=True)
    # _ = Client(n_workers=n_workers, threads_per_worker=2)
    config = OceanDatasetConfig.from_file(config)
    n_split = config.n_split
    config = config.dataset_computation
    ds = construct_lazy_dataset(config)
    standard_names = config.standard_names
    inner_chunks = config.chunking.get_chunks(standard_names)
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    ds = clear_compressors_encoding(ds)
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")
    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        n_partitions = n_split
        ds.partition.initialize_store(output_store, inner_chunks=inner_chunks)
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
    main()
