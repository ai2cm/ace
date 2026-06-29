"""
Coarsen the existing daily-mean UFS Replay zarr dataset to 5-day means.

Processes variable-by-variable to keep memory bounded and avoid building a
massive dask task graph.  The output store is initialized lazily (metadata
only) so the chunk/shard layout matches the beam pipeline convention
(time chunk=1, time shard=360, zarr v3), then each time-varying variable is
filled with a region write.

Usage:
    python coarsen_to_5day.py <input_zarr> <output_zarr> [--factor 5]

Example (GCS):
    python coarsen_to_5day.py \
    gs://vcm-ml-intermediate/2026-06-26-ufs-replay-ocean-1deg-19level-1994-2023.zarr \
    gs://vcm-ml-intermediate/2026-06-28-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr
"""

import argparse
import logging
import time

import dask.array as darr
import numpy as np
import xarray as xr
import zarr

TIME_INVARIANT_PREFIXES = ("mask_", "idepth_")
TIME_INVARIANT_NAMES = {"land_fraction", "sea_surface_fraction", "deptho", "mask_2d"}

# Match the beam pipeline output layout (see run-dataflow.sh:
# --output_time_chunksize 1 --output_time_shardsize 360).
TIME_CHUNK = 1
TIME_SHARD = 360

BATCH_SIZE = 500  # input timesteps to load at a time (divisible by factor)

KEEP_COORDS = {"time", "lat", "lon"}


def is_time_invariant(name: str) -> bool:
    if name in TIME_INVARIANT_NAMES:
        return True
    return any(name.startswith(p) for p in TIME_INVARIANT_PREFIXES)


def _drop_stray_coords(ds: xr.Dataset) -> xr.Dataset:
    """Drop non-dimension coords (cftime, ftime, z_l, …) that otherwise
    leak into the output encoding as a 'coordinates' attribute."""
    stray = [c for c in ds.coords if c not in KEEP_COORDS and c not in ds.dims]
    return ds.drop_vars(stray) if stray else ds


def coarsen_dataset(input_path: str, output_path: str, factor: int = 5):
    logging.info("Opening %s", input_path)
    ds = xr.open_zarr(input_path)
    ds = _drop_stray_coords(ds)

    n_times = ds.sizes["time"]
    nlat = ds.sizes["lat"]
    nlon = ds.sizes["lon"]
    logging.info(
        "Input: %d timesteps, %d lat, %d lon, %d vars",
        n_times,
        nlat,
        nlon,
        len(ds.data_vars),
    )

    usable = (n_times // factor) * factor
    n_out = usable // factor
    remainder = n_times - usable
    if remainder > 0:
        logging.warning("Trimming %d trailing timesteps", remainder)

    # Separate time-varying from time-invariant variables
    time_vars = []
    invariant_vars = []
    for name in ds.data_vars:
        if is_time_invariant(name) or "time" not in ds[name].dims:
            invariant_vars.append(name)
        else:
            time_vars.append(name)
    logging.info(
        "Time-varying: %d, time-invariant: %d", len(time_vars), len(invariant_vars)
    )

    # Output time coordinate: average each group of `factor` daily steps,
    # matching how the beam pipeline derives its time labels.  For an odd
    # factor this lands on the middle day (preserving the 09Z label).
    out_times = (
        ds["time"]
        .isel(time=slice(0, usable))
        .coarsen(time=factor, boundary="trim")
        .mean()
        .values
    )

    # --- Initialize the output store (metadata + coords + invariant vars) ---
    # Time-varying variables are backed by lazy dask zeros and written with
    # compute=False, so the zeros are never materialized; we overwrite each
    # variable's full time range with a region write below.
    time_shard = min(TIME_SHARD, n_out)
    logging.info(
        "Initializing output store at %s (chunk=%d, shard=%d in time)",
        output_path,
        TIME_CHUNK,
        time_shard,
    )

    skeleton_vars = {}
    encoding = {}
    for name in time_vars:
        skeleton_vars[name] = (
            ["time", "lat", "lon"],
            darr.zeros(
                (n_out, nlat, nlon),
                chunks=(time_shard, nlat, nlon),
                dtype=np.float32,
            ),
            dict(ds[name].attrs),
        )
        encoding[name] = {
            "chunks": (TIME_CHUNK, nlat, nlon),
            "shards": (time_shard, nlat, nlon),
            "dtype": "float32",
        }
    for name in invariant_vars:
        da = ds[name].load()
        da.encoding = {}
        skeleton_vars[name] = da

    skeleton = xr.Dataset(
        skeleton_vars,
        coords={"time": out_times, "lat": ds.lat, "lon": ds.lon},
    )
    skeleton.to_zarr(
        output_path,
        mode="w",
        encoding=encoding,
        compute=False,
        zarr_format=3,
        consolidated=False,
    )

    # --- Fill each time-varying variable with a region write ---
    t_start = time.time()
    batch = max(factor, (BATCH_SIZE // factor) * factor)
    logging.info(
        "Processing %d vars, batch_size=%d input timesteps", len(time_vars), batch
    )

    for vi, name in enumerate(time_vars):
        var_start = time.time()
        out_array = np.empty((n_out, nlat, nlon), dtype=np.float32)
        out_idx = 0

        for t0 in range(0, usable, batch):
            t1 = min(t0 + batch, usable)
            chunk = ds[name].isel(time=slice(t0, t1)).values  # eager load
            n_groups = (t1 - t0) // factor
            reshaped = chunk[: n_groups * factor].reshape(n_groups, factor, nlat, nlon)
            out_array[out_idx : out_idx + n_groups] = reshaped.mean(axis=1)
            out_idx += n_groups

        # Region write of the full time range writes complete shards.
        xr.Dataset({name: (["time", "lat", "lon"], out_array)}).to_zarr(
            output_path,
            region={"time": slice(0, n_out)},
        )

        logging.info(
            "  [%d/%d] %s done (%.1fs, wrote %d timesteps)",
            vi + 1,
            len(time_vars),
            name,
            time.time() - var_start,
            out_idx,
        )

    zarr.consolidate_metadata(output_path)

    total = time.time() - t_start
    logging.info(
        "Done in %.1f minutes. Output: %d timesteps at %s",
        total / 60,
        n_out,
        output_path,
    )

    print("\n=== Summary ===")
    print(f"Input:  {n_times} daily timesteps")
    print(f"Output: {n_out} {factor}-day mean timesteps")
    print(f"Time range: {out_times[0]} to {out_times[-1]}")
    print(f"Total time: {total / 60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Coarsen daily UFS zarr to N-day means"
    )
    parser.add_argument("input_zarr", help="Path to input daily zarr store")
    parser.add_argument("output_zarr", help="Path to output coarsened zarr store")
    parser.add_argument(
        "--factor",
        type=int,
        default=5,
        help="Number of days to average (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    coarsen_dataset(args.input_zarr, args.output_zarr, args.factor)


if __name__ == "__main__":
    main()
