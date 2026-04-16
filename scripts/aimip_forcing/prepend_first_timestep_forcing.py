import logging

import click
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from encoding import clear_encoding, set_shards_chunks

# this script is based on the notebook at
# https://github.com/ai2cm/explore2/blob/main/brianh/2025-06-18-ERA5-SHiELD-AMIP-forcing/2025-11-24-repeat-first-timestep-AIMIP-forcing.ipynb

INPUT_FORCING_PATH = (
    "gs://vcm-ml-intermediate/2025-09-09-aimip-era5-1deg-forcing-1978-2024.zarr"
)
INPUT_TIMESTAMP = "1978-10-01T00:00:00"
OUTPUT_TIMESTAMP = "1978-09-30T18:00:00"


def write_output_zarr(ds: xr.Dataset, output_data_file: str):
    """
    Write the output dataset to a zarr file, one variable at a time to avoid
    excessive memory usage that seems to be a problem with the dask graph.
    """
    initial = True
    for var in ds.data_vars:
        mode = "w" if initial else "a"
        zarr_write = ds[[var]].to_zarr(
            output_data_file, mode=mode, compute=False, consolidated=False
        )
        with ProgressBar():
            zarr_write.compute()
        initial = False


@click.command()
@click.argument("local_output_filepath", type=click.Path())
@click.option(
    "--input-forcing-path",
    type=str,
    default=INPUT_FORCING_PATH,
    help="Path to existing AIMIP forcing zarr (GCS or local).",
)
@click.option(
    "--input-timestamp",
    type=str,
    default=INPUT_TIMESTAMP,
    help="Timestamp of the first forcing step to repeat.",
)
@click.option(
    "--output-timestamp",
    type=str,
    default=OUTPUT_TIMESTAMP,
    help="New timestamp to assign to the prepended step.",
)
def main(
    local_output_filepath: str,
    input_forcing_path: str,
    input_timestamp: str,
    output_timestamp: str,
):
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Opening AIMIP forcing data from {input_forcing_path}")
    ds = xr.open_zarr(input_forcing_path, chunks={"time": 100})

    logging.info(
        f"Prepending timestep {input_timestamp} relabeled as {output_timestamp}"
    )
    first_step = ds.sel(time=[input_timestamp]).assign_coords(
        time=[np.datetime64(output_timestamp)]
    )
    ds_with_prepended = xr.concat([first_step, ds], dim="time")

    logging.info("Setting chunking and sharding for output.")
    ds_with_prepended = clear_encoding(ds_with_prepended)
    ds_with_prepended = set_shards_chunks(ds_with_prepended)

    logging.info(f"Writing output to {local_output_filepath}")
    logging.info(ds_with_prepended.info())
    write_output_zarr(ds_with_prepended, local_output_filepath)


if __name__ == "__main__":
    main()
