import argparse
import logging

import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam
from apache_beam.pipeline import PipelineOptions


def get_template(ds, target_chunks) -> xr.Dataset:
    """
    Separate variables into time dependent and time invariant variables.
    Create a template dataset with the time dependent variables expanded to the
    full time dimension and the time invariant variables as is.  The time
    invariant data will be uploaded directly when the zarr is initialized.
    """
    time_dim = "time"
    time_vars = [v for v in ds.data_vars if time_dim in ds[v].dims]
    invariant = [v for v in ds.data_vars if time_dim not in ds[v].dims]

    temporal = ds[time_vars].isel({time_dim: 0}).drop_vars([time_dim])
    temporal = (
        xbeam.make_template(temporal)
        .expand_dims({time_dim: len(ds.time)})
        .chunk(target_chunks)
    )
    temporal = temporal.assign_coords({time_dim: ds.time})
    invariant = ds[invariant].load()
    return xr.merge([temporal, invariant]), ds[time_vars]


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=str, help="Path to zarr store to rechunk")
    parser.add_argument(
        "destination_path", type=str, help="Path to zarr store to write to"
    )
    parser.add_argument("time_chunk_size", type=int, help="Size of time chunks")
    return parser


def main():
    args, pipeline_args = _get_parser().parse_known_args()

    ds, chunks = xbeam.open_zarr(args.source_path)
    # Only supports adjusting time chunk size for now
    time_chunk = {"time": args.time_chunk_size}
    store = args.destination_path
    template, to_rechunk_ds = get_template(ds, time_chunk)

    logging.basicConfig(level=logging.INFO)

    recipe = (
        xbeam.DatasetToChunks(to_rechunk_ds, chunks, split_vars=True)
        | xbeam.SplitChunks(time_chunk)
        | xbeam.ConsolidateChunks(time_chunk)
        | xbeam.ChunksToZarr(store, template, time_chunk)
    )

    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        p | recipe


if __name__ == "__main__":
    main()
