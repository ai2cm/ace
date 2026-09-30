"""Compute only the variogram score scaling (variogram-scaling.nc) of existing
dataset stores, merged into one file.

Each variable is taken from the first input store that has it, matching
merge_stats.py's first-wins rule, so a later store only computes the variables
the earlier ones lack.

Example:

    python get_variogram_scaling.py \
        --input gs://bucket/store-a.zarr --input gs://bucket/store-b.zarr \
        --start-date 1990-01-01 --end-date 2019-12-31 \
        --output gs://bucket/stats-dir/variogram-scaling.nc
"""

import logging
import os
import tempfile
import time

import click
import xarray as xr
from get_stats import (
    DROP_VARIABLES,
    LAT_LON_DIMS,
    add_history_attrs,
    compute_variogram_scaling,
    copy,
)

# 360 daily steps is one shard of the ERA5 daily stores; reading whole shards
# avoids re-reading a shard per chunk
DEFAULT_TIME_CHUNK = 360


def get_merged_variogram_scaling(
    inputs: list[str],
    lat_dim: str,
    lon_dim: str,
    start_date: str | None,
    end_date: str | None,
    window_size: int = 5,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    storage_options: dict | None = None,
) -> xr.Dataset:
    """The variogram scaling of each variable of the first input store that has
    it, computed over [start_date, end_date]."""
    try:
        import dask  # noqa: F401

        chunks: dict | None = {"time": time_chunk}
    except ImportError:
        logging.warning("Could not import dask, chunking is disabled.")
        chunks = None
    merged: list[xr.Dataset] = []
    done: set[str] = set()
    n_samples = []
    for path in inputs:
        ds = xr.open_zarr(path, chunks=chunks, storage_options=storage_options)
        ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
        ds = ds.drop_vars([name for name in ds.data_vars if name in done])
        if "time" in ds.dims:
            ds = ds.sel(time=slice(start_date, end_date))
            n_samples.append(len(ds.time))
        if len(ds.data_vars) == 0:
            logging.info(f"No new variables in {path}, skipping")
            continue
        logging.info(f"Computing variogram scaling of {len(ds.data_vars)} from {path}")
        t0 = time.time()
        result = compute_variogram_scaling(
            ds, lat_dim=lat_dim, lon_dim=lon_dim, window_size=window_size
        ).compute()
        logging.info(f"Computed in {time.time() - t0:0.1f} s")
        merged.append(result)
        done.update(result.data_vars)
    if len(merged) == 0:
        raise ValueError(f"No lat-lon variables found in {inputs}")
    out = xr.merge(merged, combine_attrs="override")
    add_history_attrs(
        out, ", ".join(inputs), start_date, end_date, n_samples[0] if n_samples else 0
    )
    return out


@click.command()
@click.option(
    "--input",
    "inputs",
    multiple=True,
    required=True,
    help="Input zarr store; repeat for several (first store with a variable wins).",
)
@click.option("--output", required=True, help="Output netCDF path (local or remote).")
@click.option(
    "--data-type",
    type=click.Choice(sorted(LAT_LON_DIMS)),
    default="ERA5",
    show_default=True,
)
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
@click.option("--time-chunk", default=DEFAULT_TIME_CHUNK, show_default=True)
@click.option(
    "--gcs-token",
    default=None,
    help="gcsfs token, e.g. 'cloud' to use the VM's service account.",
)
def main(
    inputs: tuple[str, ...],
    output: str,
    data_type: str,
    start_date: str | None,
    end_date: str | None,
    time_chunk: int,
    gcs_token: str | None,
):
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    lat_dim, lon_dim = LAT_LON_DIMS[data_type]
    storage_options = {"token": gcs_token} if gcs_token is not None else None
    ds = get_merged_variogram_scaling(
        list(inputs),
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        start_date=start_date,
        end_date=end_date,
        time_chunk=time_chunk,
        storage_options=storage_options,
    )
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "variogram-scaling.nc")
        ds.to_netcdf(local)
        if gcs_token is not None and output.startswith("gs://"):
            import gcsfs

            gcsfs.GCSFileSystem(token=gcs_token).put(local, output)
        else:
            copy(local, output)
    logging.info(f"Wrote {output} in {time.time() - t0:0.1f} s")


if __name__ == "__main__":
    main()
