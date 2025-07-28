import os
import pdb
import sys

import click
import dask
import xarray as xr
from dask.distributed import Client

DIMS = ["time", "lat", "lon"]


def add_history_attrs(
    ds: xr.Dataset, input_zarr: str, n_samples: int, start_date: str, end_date: str
):
    ds.attrs["history"] = (
        f"INPUT_ZARR: {input_zarr}, START_DATE: {start_date}, END_DATE: {end_date}."
    )
    ds.attrs["input_samples"] = n_samples


def get_stats(
    input_ds: xr.Dataset,
    stats_type="centering",
):
    print(f"Computing {stats_type} stats")
    if stats_type == "centering":
        stats_ds = input_ds.mean(dim=DIMS).compute()
    elif stats_type == "scaling-full-field":
        stats_ds = input_ds.std(dim=DIMS).compute()
    elif stats_type == "scaling-residual":
        stats_ds = input_ds.diff("time").std(dim=DIMS).compute()
    elif stats_type == "time-mean":
        stats_ds = input_ds.mean(dim="time").compute()
    else:
        raise ValueError(f"Unrecognized stats {stats_type}")
    return stats_ds


@click.command()
@click.option("--input_zarr", help="Path to dataset zarr.")
@click.option("--stats_dir", help="Path to dataset zarr.")
@click.option(
    "--start",
    help="Start datetime in format 'YYYY-MM-DD HH:MM:SS'",
)
@click.option(
    "--end",
    help="End datetime in format 'YYYY-MM-DD HH:MM:SS'",
)
@click.option(
    "--n_dask_workers",
    type=int,
)
def main(
    input_zarr,
    stats_dir,
    start: str | None,
    end: str | None,
    n_dask_workers: int | None,
):
    if n_dask_workers is not None:
        print(f"Using dask Client(n_workers={n_dask_workers})...")
        client = Client(n_workers=n_dask_workers)
        print(client.dashboard_link)
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)
    print(f"Opening {input_zarr}")
    with dask.config.set({"array.chunk-size": "128MiB"}):
        input_ds = xr.open_zarr(input_zarr, chunks={"time": "auto"})
    if start is None:
        start = input_ds["time"].values[0]
    if end is None:
        end = input_ds["time"].values[-1]
    assert start is not None
    assert end is not None
    time_slice = slice(start, end)
    input_ds = input_ds.sel(time=time_slice)
    n_samples = len(input_ds.time)
    for stats_type in [
        "centering",
        "scaling-full-field",
        "scaling-residual",
        "time-mean",
    ]:
        stats_ds = get_stats(
            input_ds=input_ds,
            stats_type=stats_type,
        )
        add_history_attrs(
            stats_ds,
            input_zarr,
            n_samples,
            start,
            end,
        )
        stats_path = os.path.join(stats_dir, f"{stats_type}.nc")
        stats_ds.to_netcdf(stats_path)
        print(f"Wrote {stats_type} stats to {stats_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        pdb.post_mortem()  # Start the debugger
        raise  # Re-raise the exception to preserve the traceback
