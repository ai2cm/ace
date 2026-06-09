"""Stage a local netCDF slice of the ERA5 4-deg data for the blowup rollout.

The investigation box has no /climate-default (weka) mount and its forkserver
multiprocessing is broken, so the fme forcing loader cannot stream the zarr from
GCS (it forces a forkserver worker whenever the zarr engine is used). We sidestep
both problems by downloading exactly the variables and time range the rollout
needs into a single local netCDF, then pointing the inference config at it with
the h5netcdf engine and num_data_workers=0 (a pure single-process code path).

The variable set is read from the checkpoint so it stays in sync with the model.
"""

import argparse
import json
import os

import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

GCS_ZARR = (
    "gs://vcm-ml-intermediate/2026-04-17-era5-4deg-8layer-daily-1940-2025/"
    "2026-03-19-era5-4deg-8layer-1940-2025.zarr"
)
# Resolve defaults relative to this script so staging always lands inside the
# scripts folder regardless of the caller's working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoint", "best_inference_ckpt.tar")
DEFAULT_OUT = os.path.join(HERE, "data", "era5_4deg_blowup_slice.nc")
DEFAULT_START = "1979-01-01"


def needed_variables(checkpoint_path: str) -> list[str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = ckpt["stepper"]["config"]["step"]["config"]
    names = set(step["in_names"]) | set(step["out_names"])
    names |= set(step.get("next_step_forcing_names", []))
    ocean = step.get("ocean") or {}
    for key in ("surface_temperature_name", "ocean_fraction_name"):
        if ocean.get(key):
            names.add(ocean[key])
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--start", default=DEFAULT_START, help="IC date (YYYY-MM-DD)")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2922,  # ~8 years of daily steps
        help="Number of forward steps; we stage n_steps+1 daily timesteps.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    names = needed_variables(args.checkpoint)
    print(f"Staging {len(names)} variables from {args.start} for {args.n_steps} steps")

    ds = xr.open_dataset(
        GCS_ZARR,
        engine="zarr",
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
        chunks={},
    )
    time_index = xr.CFTimeIndex(ds.time.values)
    start = time_index.get_loc(args.start)
    if isinstance(start, slice):
        start = start.start
    stop = start + args.n_steps + 1
    print(
        f"time slice {start}:{stop} -> "
        f"{ds.time.values[start]} .. {ds.time.values[stop - 1]}"
    )

    keep = [n for n in names if n in ds]
    missing = [n for n in names if n not in ds]
    if missing:
        print(f"WARNING: variables missing from source and skipped: {missing}")
    sub = ds[keep].isel(time=slice(start, stop)).load()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 1} for v in sub.data_vars}
    sub.to_netcdf(args.out, engine="h5netcdf", encoding=encoding)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size_mb:.0f} MB), dims={dict(sub.sizes)}")

    meta = {
        "source": GCS_ZARR,
        "start": args.start,
        "n_steps": args.n_steps,
        "n_times": int(sub.sizes["time"]),
        "variables": keep,
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
