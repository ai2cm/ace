# This script is used to compute a training dataset from the "raw"
# FV3GFS data stored in zarr form on GCS.

# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

# The resulting dataset is about 194GB (the input is about 2.5TB). Running this script
# on my 8-CPU VM takes about 2.5 hours. See "compute_dataset_fv3gfs_argo_workflow.yaml"
# for a workflow which parallelizes this script across the 11-member ensemble and runs
# it on our GKE cluster.

import os
import pdb
import sys
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
from functools import partial

import click
import earth2grid
import numpy as np
import torch
import xarray as xr
import xpartition  # noqa
from compute_dataset import (
    LATENT_HEAT_OF_VAPORIZATION,
    DatasetComputationConfig,
    DatasetConfig,
    DLWPChunkingConfig,
    DLWPNameMapping,
    assert_column_integral_of_moisture_is_conserved,
    assert_global_dry_air_mass_conservation,
    assert_global_moisture_conservation,
    get_dataset_urls,
    open_datasets,
)


def regrid_tensor(x, regrid_func, shape):
    data = regrid_func(torch.tensor(x, dtype=torch.double))
    return data.numpy().reshape(shape)


def _pool_func(ds, store, n_partitions, partition_dims, i):
    ds.partition.write(store, n_partitions, partition_dims, i)
    print(f"Finished writing partition {i}")


def hpx_regrid(
    ds: xr.Dataset,
    dlwp_names: DLWPNameMapping,
    level: int = 6,  # regrid resolution
    n_side: int = 64,
) -> Tuple[xr.Dataset, xr.Dataset]:
    lat_long_names = dlwp_names.lat_lon_dims
    longitude = lat_long_names[1]
    latitude = lat_long_names[0]
    lons = ds[longitude]
    lats = ds[latitude]

    hpx = earth2grid.healpix.Grid(
        level=level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )
    src = earth2grid.latlon.LatLonGrid(lat=list(lats), lon=list(lons))
    # Regridder
    regrid = earth2grid.get_regridder(src, hpx)

    ds_regridded = xr.apply_ufunc(
        regrid_tensor,
        ds,
        input_core_dims=[[latitude, longitude]],
        output_core_dims=[["face", "height", "width"]],
        output_sizes={"face": 12, "height": n_side, "width": n_side},
        output_dtypes=[float],
        dask="parallelized",
        vectorize=True,
        on_missing_core_dim="copy",
        kwargs={"regrid_func": regrid, "shape": (12, n_side, n_side)},
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    # Assign coordinates to the regridded dataset
    time_coords = ds.coords["time"]
    nside_coords = np.arange(n_side)
    grid_coords = np.arange(12)
    ds_regridded = ds_regridded.assign_coords(
        time=time_coords,
        face=grid_coords,
        height=nside_coords,
        width=nside_coords,
    )

    return ds_regridded


def construct_hpx_dataset(
    config: DatasetComputationConfig,
    run_directory: str,
    output_directory: str,
    toy_dataset: bool = False,
) -> xr.Dataset:
    dlwp_names = config.standard_names
    if not isinstance(dlwp_names, DLWPNameMapping):
        raise TypeError("Expected to be passed type of DLWPNameMapping.")
    dlwp_chunking = config.chunking
    if not isinstance(dlwp_chunking, DLWPChunkingConfig):
        raise TypeError("Expected to be passed type of DLWPChunkingConfig.")

    urls = get_dataset_urls(config, run_directory)
    print(urls)
    ds = open_datasets(config, urls)
    for var in ds:
        del ds[var].encoding["chunks"]
        del ds[var].encoding["preferred_chunks"]
    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    if toy_dataset:
        ds = ds.isel(time=slice(0, 200))
    # We would like to:

    # 1. map to healpix mesh
    ds = hpx_regrid(
        ds=ds,
        dlwp_names=dlwp_names,
        n_side=64,
    )
    print(f"After regrid: {ds}")

    # 2. chunk and save
    chunks = config.chunking.get_chunks(dlwp_names)
    ds = ds.chunk(chunks)
    ds.attrs["history"] = (
        "Dataset computed by full-model/scripts/data_process"
        "/compute_hpx_dataset.py"
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
@click.option("--num-processes", default=16, help="Number of processes to spin up.")
def main(
    config,
    run_directory,
    output_store,
    debug,
    subsample,
    check_conservation,
    num_processes,
):
    config = DatasetConfig.from_file(config).dataset_computation
    dlwp_names = config.standard_names
    print(f"--run-directory is {run_directory}")
    print(f"--output-store is {output_store}")
    ds = construct_hpx_dataset(
        config=config,
        run_directory=run_directory,
        output_directory=output_store,
        toy_dataset=False,
    )
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    if check_conservation:
        assert_column_integral_of_moisture_is_conserved(
            ds,
            precipitable_water_path_name=dlwp_names.precipitable_water_path,
            total_water_path_name=dlwp_names.total_water_path,
        )
        assert_global_dry_air_mass_conservation(
            ds,
            dims=dlwp_names.horizontal_dims,
            surface_pressure_name=dlwp_names.surface_pressure,
            total_water_path_name=dlwp_names.total_water_path,
            latitude_dim=dlwp_names.latitude_dim,
            time_dim=dlwp_names.time_dim,
        )
        assert_global_moisture_conservation(
            ds,
            dims=dlwp_names.horizontal_dims,
            latitude_dim=dlwp_names.latitude_dim,
            total_water_path_name=dlwp_names.total_water_path,
            latent_heat_flux_name=dlwp_names.latent_heat_flux,
            latent_heat_of_vaporization=LATENT_HEAT_OF_VAPORIZATION,
            precip_rate_name=dlwp_names.precip_rate,
            time_dim=dlwp_names.time_dim,
        )
    drop_vars = [var for var in dlwp_names.dropped_variables if var in ds]
    ds = ds.drop(drop_vars)
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        n_partitions = config.n_split
        partition_dims = [dlwp_names.time_dim]
        store = f"{output_store}.zarr"
        ds.partition.initialize_store(store)

        with mp.get_context("forkserver").Pool(num_processes) as pool:
            pool.map(
                partial(_pool_func, ds, store, n_partitions, partition_dims),
                range(n_partitions),
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
