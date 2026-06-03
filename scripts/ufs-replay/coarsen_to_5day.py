"""
Coarsen the existing daily-mean UFS Replay zarr dataset to 5-day means.

Reads the fully processed 1-day dataset, averages every 5 consecutive
timesteps, and writes a new zarr store.  Time-invariant fields (mask_*,
idepth_*, land_fraction, sea_surface_fraction, deptho) are copied as-is.

Usage:
    python coarsen_to_5day.py <input_zarr> <output_zarr> [--factor 5]

Example (GCS):
    python coarsen_to_5day.py \
      gs://vcm-ml-intermediate/2026-06-01-ufs-replay-ocean-1deg-19level-1994-2023.zarr \
      gs://vcm-ml-intermediate/2026-06-03-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr

Example (local):
    python coarsen_to_5day.py ./ufs-daily.zarr ./ufs-5day.zarr
"""

import argparse
import logging

import numpy as np
import xarray as xr

TIME_INVARIANT_PREFIXES = ("mask_", "idepth_")
TIME_INVARIANT_NAMES = {"land_fraction", "sea_surface_fraction", "deptho", "mask_2d"}


def is_time_invariant(name: str) -> bool:
    if name in TIME_INVARIANT_NAMES:
        return True
    return any(name.startswith(p) for p in TIME_INVARIANT_PREFIXES)


def coarsen_dataset(input_path: str, output_path: str, factor: int = 5):
    logging.info("Opening %s", input_path)
    ds = xr.open_zarr(input_path)

    n_times = ds.sizes["time"]
    logging.info("Input: %d timesteps, %d variables", n_times, len(ds.data_vars))

    usable = (n_times // factor) * factor
    remainder = n_times - usable
    if remainder > 0:
        logging.warning(
            "Trimming %d trailing timesteps (%d not divisible by %d)",
            remainder,
            n_times,
            factor,
        )

    # Separate time-varying and time-invariant variables
    time_vars = []
    invariant_vars = []
    for name in ds.data_vars:
        if is_time_invariant(name) or "time" not in ds[name].dims:
            invariant_vars.append(name)
        else:
            time_vars.append(name)

    logging.info(
        "Time-varying: %d vars, time-invariant: %d vars",
        len(time_vars),
        len(invariant_vars),
    )

    # Coarsen time-varying variables
    ds_tv = ds[time_vars].isel(time=slice(0, usable))
    logging.info(
        "Coarsening %d timesteps by factor %d -> %d", usable, factor, usable // factor
    )
    ds_coarsened = ds_tv.coarsen(time=factor, boundary="exact").mean()

    # Snap time labels to 12Z on the middle day of each window
    snapped = []
    for t in ds_coarsened.time.values:
        if hasattr(t, "year"):
            import cftime

            if isinstance(t, cftime.datetime):
                snapped.append(type(t)(t.year, t.month, t.day, 12, 0, 0))
            else:
                snapped.append(np.datetime64(f"{t.astype('datetime64[D]')}T12:00:00"))
        else:
            snapped.append(t)
    ds_coarsened = ds_coarsened.assign_coords(time=snapped)

    # Merge with invariant fields
    ds_inv = ds[invariant_vars]
    ds_out = xr.merge([ds_coarsened, ds_inv])

    # Preserve float32
    for name in ds_out.data_vars:
        if ds_out[name].dtype == np.float64:
            ds_out[name] = ds_out[name].astype(np.float32)

    n_out = ds_out.sizes.get("time", 0)
    logging.info("Output: %d timesteps, %d variables", n_out, len(ds_out.data_vars))

    # Write — rechunk uniformly and clear source encoding to avoid conflicts
    logging.info("Writing to %s", output_path)
    chunk_spec = {}
    if "time" in ds_out.dims:
        chunk_spec["time"] = min(50, n_out)
    if "lat" in ds_out.dims:
        chunk_spec["lat"] = ds_out.sizes["lat"]
    if "lon" in ds_out.dims:
        chunk_spec["lon"] = ds_out.sizes["lon"]

    ds_out = ds_out.chunk(chunk_spec)
    for var in ds_out.data_vars:
        ds_out[var].encoding.clear()
    for coord in ds_out.coords:
        ds_out[coord].encoding.clear()
    ds_out.to_zarr(output_path, mode="w", consolidated=True)
    logging.info("Done. Output at %s", output_path)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Input:  {n_times} daily timesteps")
    print(f"Output: {n_out} {factor}-day mean timesteps")
    print(f"Time range: {ds_out.time.values[0]} to {ds_out.time.values[-1]}")
    print(f"Variables: {sorted(ds_out.data_vars)}")


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
