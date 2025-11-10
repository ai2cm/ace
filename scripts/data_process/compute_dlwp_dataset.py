# This script takes a generated hpx dataset and computes the DLWP dataset.
import math
import multiprocessing as mp
import sys
from functools import partial

import click
import numpy as np
import xarray as xr
import xpartition  # noqa
from compute_dataset import (
    DatasetComputationConfig,
    DatasetConfig,
    DLWPNameMapping,
    clear_compressors_encoding,
)


def _pool_func(ds, store, n_partitions, partition_dims, i):
    ds.partition.write(
        store, n_partitions, partition_dims, i, collect_variable_writes=True
    )
    print(f"Finished writing partition {i}", flush=True)


def infer_channel_vars(ds):
    channel_vars = [
        var
        for var in ds.data_vars
        if set(ds[var].dims) == {"time", "face", "height", "width"}
    ]
    return channel_vars


def convert_era5_data_to_dlwp_data(
    ds, chunks
):  # NOTE: ds must already be regridded to the hpx grid
    # 1. Identify constants and channel variables
    constants = [
        var for var in ds.data_vars if set(ds[var].dims) == {"face", "height", "width"}
    ]
    channel_vars = infer_channel_vars(ds)

    # Convert string coordinates to np.dtypes.StringDType for compatibility
    # with zarr v3:
    # https://github.com/pydata/xarray/issues/10077#issuecomment-2702317504
    constants = np.array(constants, dtype=np.dtypes.StringDType)
    channel_vars = np.array(channel_vars, dtype=np.dtypes.StringDType)

    print("Starting to process data")
    # Process constants once
    constant_data = xr.concat([ds[var] for var in constants], dim="channel_c")
    constant_data = constant_data.assign_coords(channel_c=constants)

    print("Finished processing constant data")
    constants_chunks = {
        "channel_c": chunks.get("channel_c", -1),
        "face": chunks.get("face", -1),
        "height": chunks.get("height", -1),
        "width": chunks.get("width", -1),
    }
    constant_data = constant_data.chunk(constants_chunks)

    # Use chunk size from chunking config, if it exists
    channel_data = xr.concat([ds[var] for var in channel_vars], dim="channel_in")
    channel_data = channel_data.transpose(
        "time", "channel_in", "face", "height", "width"
    )
    channel_data = channel_data.assign_coords(channel_in=channel_vars)

    print("Finished processing channel data")
    # Add channel dimension with actual size to chunks
    channel_chunks = chunks.copy()
    channel_chunks["channel_in"] = chunks.get("channel_in", -1)
    channel_data = channel_data.chunk(channel_chunks)

    print(type(channel_data.data))  # Should be dask.array.core.Array
    print(channel_data.chunks)

    # Build the final dataset
    ds_restructured = xr.Dataset(
        coords={
            "time": ds["time"],
            "face": ds["face"],
            "height": ds["height"],
            "width": ds["width"],
            "channel_c": ("channel_c", constants),
            "channel_in": ("channel_in", channel_vars),
            "channel_out": ("channel_out", channel_vars),
        },
        data_vars={
            "constants": (("channel_c", "face", "height", "width"), constant_data.data),
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                channel_data.data,
            ),
            "targets": (
                ("time", "channel_out", "face", "height", "width"),
                channel_data.data,
            ),
        },
    )

    # get hpx lat/lon grid
    hpx_lat = ds.lat.data.reshape((12, 64, 64))
    hpx_lon = ds.lon.data.reshape((12, 64, 64))

    final_ds = ds_restructured.assign_coords(
        lat=(("face", "height", "width"), hpx_lat),
        lon=(("face", "height", "width"), hpx_lon),
    )
    print(f"final_ds created: {final_ds}")
    print("chunks are: ", final_ds["inputs"].chunks)  # Should show your chunk sizes
    return final_ds


def infer_time_shard_size(input_store: str, time_shard_size: int) -> int:
    """Scale the configured shard size down by the number of channel variables,
    since we eventually stack all of those variables together.
    """
    ds = xr.open_zarr(input_store, chunks=None)
    channel_vars = infer_channel_vars(ds)
    num_channel_vars = len(channel_vars)
    return math.ceil(time_shard_size / num_channel_vars)


def construct_dlwp_dataset(
    config: DatasetComputationConfig,
    run_directory: str,
    output_directory: str,
    toy_dataset: bool = False,
) -> xr.Dataset:
    dlwp_names = config.standard_names
    time_dim = dlwp_names.time_dim
    if not isinstance(dlwp_names, DLWPNameMapping):
        raise TypeError("Expected to be passed type of DLWPNameMapping.")

    input_store = f"file://{run_directory}.zarr"

    # 1. massage it to be like the dlwp data
    if config.sharding is None:
        chunks = config.chunking.get_chunks(dlwp_names)
    else:
        chunks = config.sharding.get_chunks(dlwp_names)
        chunks[time_dim] = infer_time_shard_size(input_store, chunks[time_dim])

    ds = xr.open_zarr(input_store, chunks=chunks, decode_timedelta=True)

    print(f"Input dataset size is {ds.nbytes / 1e9} GB")
    if toy_dataset:
        ds = ds.isel(time=slice(0, 200))

    ds_converted = convert_era5_data_to_dlwp_data(ds, chunks)

    print(f"ds_converted created: {ds_converted}")

    # 2. save it
    ds_converted.attrs["history"] = (
        "Dataset computed by full-model/scripts/data_process"
        "/compute_dlwp_dataset.py"
        f" script, using following input zarr: {input_store}."
    )
    ds_converted.attrs["vertical_coordinate"] = (
        "The pressure at level interfaces can by computed as "
        "p_i = ak_i + bk_i * PRESsfc, where PRESsfc is the surface pressure and the "
        "p_i pressure corresponds to the interface at the top of the i'th finite "
        "volume layer, counting down from the top of atmosphere."
    )
    return ds_converted


@click.command()
@click.option("--config", help="Path to dataset configuration YAML file.")
@click.option("--run-directory", help="Path to reference run directory.")
@click.option("--output-store", help="Path to output zarr store.")
@click.option("--num-processes", default=32, help="Number of processes to spin up.")
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
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite the existing store."
)
def main(
    config,
    run_directory,
    output_store,
    num_processes,
    debug,
    subsample,
    overwrite,
):
    config = DatasetConfig.from_file(config).dataset_computation
    print(f"--run-directory is {run_directory}")
    print(f"--output-store is {output_store}")
    ds = construct_dlwp_dataset(
        config=config,
        run_directory=run_directory,
        output_directory=output_store,
        toy_dataset=False,
    )
    if subsample:
        ds = ds.isel(time=slice(10, 13))
    print(f"Output dataset size is {ds.nbytes / 1e9} GB")

    dlwp_names = config.standard_names
    if config.sharding is None:
        inner_chunks = None
    else:
        inner_chunks = config.chunking.get_chunks(dlwp_names)

    ds = clear_compressors_encoding(ds)

    store = f"{output_store}.zarr"
    if overwrite:
        # Mode "w" is used to overwrite the existing store.
        ds.partition.initialize_store(store, mode="w", inner_chunks=inner_chunks)
    else:
        try:
            ds.partition.initialize_store(store, mode="w-", inner_chunks=inner_chunks)
        except FileExistsError:
            raise ValueError(
                "Store already exists. Use --overwrite to overwrite, \
            or change config to write to a new store."
            )

    if debug:
        with xr.set_options(display_max_rows=500):
            print(ds)
    else:
        print(f"Writing dataset to {output_store}.zarr using Dask parallelism...")
        n_partitions = config.n_split
        partition_dims = [dlwp_names.time_dim]
        with mp.get_context("forkserver").Pool(num_processes) as pool:
            pool.map(
                partial(_pool_func, ds, store, n_partitions, partition_dims),
                range(n_partitions),
            )

        print(f"Finished writing dataset to {store}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        # pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
