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

# import pdb
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
    DatasetComputationConfig,
    DatasetConfig,
    DLWPChunkingConfig,
    DLWPNameMapping,
    clear_compressors_encoding,
    get_dataset_urls,
    open_datasets,
)

# After applying the regrid on 64 sides, this is always the pattern of nans.
# it includes the diagonal of face 4, and corners of 0, 1, 2, 3, 8, 9, 10, 11
NAN_INDICES = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
        [4, 1, 1],
        [4, 2, 2],
        [4, 3, 3],
        [4, 4, 4],
        [4, 5, 5],
        [4, 6, 6],
        [4, 7, 7],
        [4, 8, 8],
        [4, 9, 9],
        [4, 10, 10],
        [4, 11, 11],
        [4, 12, 12],
        [4, 13, 13],
        [4, 14, 14],
        [4, 15, 15],
        [4, 16, 16],
        [4, 17, 17],
        [4, 18, 18],
        [4, 19, 19],
        [4, 20, 20],
        [4, 21, 21],
        [4, 22, 22],
        [4, 23, 23],
        [4, 24, 24],
        [4, 25, 25],
        [4, 26, 26],
        [4, 27, 27],
        [4, 28, 28],
        [4, 29, 29],
        [4, 30, 30],
        [4, 31, 31],
        [4, 32, 32],
        [4, 33, 33],
        [4, 34, 34],
        [4, 35, 35],
        [4, 36, 36],
        [4, 37, 37],
        [4, 38, 38],
        [4, 39, 39],
        [4, 40, 40],
        [4, 41, 41],
        [4, 42, 42],
        [4, 43, 43],
        [4, 44, 44],
        [4, 45, 45],
        [4, 46, 46],
        [4, 47, 47],
        [4, 48, 48],
        [4, 49, 49],
        [4, 50, 50],
        [4, 51, 51],
        [4, 52, 52],
        [4, 53, 53],
        [4, 54, 54],
        [4, 55, 55],
        [4, 56, 56],
        [4, 57, 57],
        [4, 58, 58],
        [4, 59, 59],
        [4, 60, 60],
        [4, 61, 61],
        [4, 62, 62],
        [4, 63, 63],
        [8, 63, 63],
        [9, 63, 63],
        [10, 63, 63],
        [11, 63, 63],
    ]
)


def fill_nans_with_neighbors(arr, nan_indices):
    arr_filled = arr.copy()
    n_face, n_height, n_width = arr.shape
    for f, h, w in nan_indices:
        neighbors = []
        # Up
        if h > 0:
            neighbors.append(arr[f, h - 1, w])
        # Down
        if h < n_height - 1:
            neighbors.append(arr[f, h + 1, w])
        # Left
        if w > 0:
            neighbors.append(arr[f, h, w - 1])
        # Right
        if w < n_width - 1:
            neighbors.append(arr[f, h, w + 1])
        arr_filled[f, h, w] = np.mean(neighbors)
    return arr_filled


def regrid_tensor(x, regrid_func, shape):
    data = regrid_func(torch.tensor(x, dtype=torch.double))
    arr_hpx = data.numpy().reshape(shape)
    # replace corner nans with mean-filling
    arr_hpx = fill_nans_with_neighbors(arr_hpx, NAN_INDICES)
    return arr_hpx


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
    lat_hpx = hpx.lat  # shape (face, height, width)
    lon_hpx = hpx.lon  # shape (face, height, width)
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
    ds_regridded = ds_regridded.assign_coords(
        lat=lat_hpx.data,
        lon=lon_hpx.data,
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
    if config.sharding is None:
        outer_chunks = config.chunking.get_chunks(dlwp_names)
    else:
        outer_chunks = config.sharding.get_chunks(dlwp_names)

    ds = ds.chunk(outer_chunks)
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
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print metadata instead of writing output.",
)
@click.option(
    "--subsample",
    is_flag=True,
    default=False,
    help="Subsample the data before writing.",
)
@click.option("--num-processes", default=16, help="Number of processes to spin up.")
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite the existing store."
)
def main(
    config,
    run_directory,
    output_store,
    debug,
    subsample,
    num_processes,
    overwrite,
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
    drop_vars = [var for var in dlwp_names.dropped_variables if var in ds]
    ds = ds.drop(drop_vars)
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")

    if config.sharding is None:
        inner_chunks = None
    else:
        inner_chunks = config.chunking.get_chunks(dlwp_names)

    ds = clear_compressors_encoding(ds)

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        n_partitions = config.n_split
        partition_dims = [dlwp_names.time_dim]
        store = f"{output_store}.zarr"
        if overwrite:
            # Mode "w" is used to overwrite the existing store.
            ds.partition.initialize_store(store, mode="w", inner_chunks=inner_chunks)
        else:
            try:
                ds.partition.initialize_store(
                    store, mode="w-", inner_chunks=inner_chunks
                )
            except FileExistsError:
                raise ValueError(
                    "Store already exists. Use --overwrite to overwrite, \
                or change config to write to a new store."
                )

        print("Initialized store, now writing partitions")
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
        # pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
