"""
Coarsen the existing daily-mean UFS Replay zarr dataset to 5-day means.

Processes variable-by-variable to keep memory bounded and avoid
building a massive dask task graph.

Usage:
    python coarsen_to_5day.py <input_zarr> <output_zarr> [--factor 5]

Example (GCS):
    python coarsen_to_5day.py \
        gs://vcm-ml-intermediate/ufs-replay-ocean-1deg-19level-1994-2023.zarr \
        gs://vcm-ml-intermediate/2026-06-03-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr
"""

import argparse
import logging
import time

import numpy as np
import xarray as xr
import zarr

TIME_INVARIANT_PREFIXES = ("mask_", "idepth_")
TIME_INVARIANT_NAMES = {"land_fraction", "sea_surface_fraction", "deptho", "mask_2d"}

BATCH_SIZE = 500  # timesteps to load at a time (must be divisible by factor)


def is_time_invariant(name: str) -> bool:
    if name in TIME_INVARIANT_NAMES:
        return True
    return any(name.startswith(p) for p in TIME_INVARIANT_PREFIXES)


def coarsen_dataset(input_path: str, output_path: str, factor: int = 5):
    logging.info("Opening %s", input_path)
    ds = xr.open_zarr(input_path)

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

    # Separate variables
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

    # Compute output time coordinate
    # Load input times and average each group of `factor`
    in_times = ds.time.values[:usable]
    out_times = []
    for i in range(0, usable, factor):
        t = in_times[i + factor // 2]  # middle day of window
        if hasattr(t, "year"):
            import cftime

            if isinstance(t, cftime.datetime):
                out_times.append(type(t)(t.year, t.month, t.day, 12, 0, 0))
            else:
                day = np.datetime64(t, "D")
                out_times.append(np.datetime64(str(day) + "T12:00:00"))
        else:
            out_times.append(t)

    # Initialize output zarr store with a skeleton dataset
    logging.info("Initializing output store at %s", output_path)
    time_chunk = min(50, n_out)

    skeleton_vars = {}
    for name in time_vars:
        skeleton_vars[name] = xr.DataArray(
            data=np.empty((0, nlat, nlon), dtype=np.float32),
            dims=["time", "lat", "lon"],
            attrs=ds[name].attrs,
        )
    for name in invariant_vars:
        skeleton_vars[name] = ds[name].load()

    skeleton = xr.Dataset(skeleton_vars)
    skeleton = skeleton.assign_coords(
        time=xr.DataArray([], dims="time"),
        lat=ds.lat,
        lon=ds.lon,
    )

    # Write skeleton to create the store structure, then resize
    encoding = {}
    for name in time_vars:
        encoding[name] = {
            "chunks": (time_chunk, nlat, nlon),
            "dtype": "float32",
        }

    skeleton.to_zarr(output_path, mode="w", encoding=encoding, consolidated=False)

    # Now open the store and resize + write the time coordinate
    store = zarr.open(output_path, mode="r+")
    for name in time_vars:
        store[name].resize(n_out, nlat, nlon)
    # Write time coordinate
    if "time" in store:
        store["time"].resize(n_out)
    else:
        store.create_dataset("time", shape=(n_out,), dtype=object, overwrite=True)
    store["time"][:] = np.array(out_times)

    # Process time-varying variables one at a time
    t_start = time.time()
    batch = max(factor, (BATCH_SIZE // factor) * factor)
    logging.info(
        "Processing %d vars, batch_size=%d input timesteps", len(time_vars), batch
    )

    for vi, name in enumerate(time_vars):
        var_start = time.time()
        out_idx = 0

        for t0 in range(0, usable, batch):
            t1 = min(t0 + batch, usable)
            chunk = ds[name].isel(time=slice(t0, t1)).values  # eager load
            n_chunk = t1 - t0
            n_groups = n_chunk // factor

            # Reshape to (n_groups, factor, lat, lon) and mean
            reshaped = chunk[: n_groups * factor].reshape(n_groups, factor, nlat, nlon)
            coarsened = reshaped.mean(axis=1).astype(np.float32)

            store[name][out_idx : out_idx + n_groups] = coarsened
            out_idx += n_groups

        elapsed = time.time() - var_start
        logging.info(
            "  [%d/%d] %s done (%.1fs, wrote %d timesteps)",
            vi + 1,
            len(time_vars),
            name,
            elapsed,
            out_idx,
        )

    # Consolidate metadata
    zarr.consolidate_metadata(output_path)

    total = time.time() - t_start
    logging.info(
        "Done in %.1f minutes. Output: %d timesteps at %s",
        total / 60,
        n_out,
        output_path,
    )

    print(f"\n=== Summary ===")
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
