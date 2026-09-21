"""Write the masked, per-land-area snow channels for one dataset to a zarr store.

The parent store is never modified. The output holds four variables with the
time coordinate copied from the parent, so a training config can attach it to
the parent with the data loader's ``merge`` key:

  surface_snow_amount_masked          per-land-area SWE, NaN where mask == 0
  surface_snow_area_fraction_masked   cover fraction in [0, 1], NaN where mask == 0
  mask_surface_snow_amount_masked         static validity mask (1/0)
  mask_surface_snow_area_fraction_masked  static validity mask (1/0)

NaN in the channels is load-bearing: the loss zeroes prediction and target
where the target is NaN and the output masker NaN-fills predictions outside the
mask, so store NaNs and mask zeros must agree. The transform is
``masked_snow.land_snow_fields``, shared with the stats script.

Chunking matches the parents: inner time chunk 1, shard 360.

Usage:
  python build_masked_snow_channels.py era5 [--dev]
  python build_masked_snow_channels.py cm4  [--dev]

Writes ./store-out/<parent-name>-land-snow-masked.zarr, resuming a partial
store if present. run_data_pipeline.sh uploads it next to the parent.
"""

import argparse
import os

import numpy as np
import xarray as xr
from masked_snow import (
    MASKED,
    PARENTS,
    SCF,
    SHARD_STEPS,
    SWE,
    land_fraction,
    land_snow_fields,
    load_mask,
    open_parent,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(PARENTS))
    parser.add_argument("--dev", action="store_true", help="first 2 shards only")
    args = parser.parse_args()
    parent = PARENTS[args.dataset]

    mask = load_mask()
    valid = mask > 0.5
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "store-out", f"{parent.output_name}.zarr")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = open_parent(parent)
    land_frac = land_fraction(ds)
    n = ds.sizes["time"] if not args.dev else 2 * SHARD_STEPS
    dims = ds[SWE].dims
    spatial_dims = dims[1:]
    spatial_coords = {d: ds[d].values for d in spatial_dims}

    static = xr.Dataset(
        {
            f"mask_{MASKED[SWE]}": (spatial_dims, mask.astype(np.float32)),
            f"mask_{MASKED[SCF]}": (spatial_dims, mask.astype(np.float32)),
        },
        coords=spatial_coords,
    )
    encoding = {
        MASKED[v]: {"chunks": (1, *mask.shape), "shards": (SHARD_STEPS, *mask.shape)}
        for v in (SWE, SCF)
    }
    attrs = {
        MASKED[SWE]: {
            "long_name": "Surface snow amount per unit land area",
            "units": "kg/m**2",
        },
        MASKED[SCF]: {
            "long_name": "Surface snow area fraction of the land area",
            "units": "fraction",
        },
    }

    done = 0
    if os.path.exists(out):
        done = xr.open_zarr(out).sizes["time"]
        print(f"  resuming: {done}/{n} steps already written", flush=True)
    refresh_every = 20 * SHARD_STEPS
    for start in range(done, n, SHARD_STEPS):
        if start > done and start % refresh_every == 0:
            ds = open_parent(parent)
        stop = min(start + SHARD_STEPS, n)
        block = ds[[SWE, SCF]].isel(time=slice(start, stop)).load()
        swe, scf = land_snow_fields(
            block[SWE].values, block[SCF].values, parent, land_frac, valid
        )
        masked = xr.Dataset(
            {
                MASKED[SWE]: (dims, swe, attrs[MASKED[SWE]]),
                MASKED[SCF]: (dims, scf, attrs[MASKED[SCF]]),
            },
            coords={"time": block["time"].values, **spatial_coords},
        )
        if start == 0:
            masked.merge(static).to_zarr(out, mode="w", encoding=encoding)
        else:
            masked.to_zarr(out, mode="a", append_dim="time")
        print(f"  {args.dataset}: {stop}/{n} steps", flush=True)

    check = xr.open_zarr(out)
    assert check["time"].equals(ds["time"].isel(time=slice(0, n)))
    first = check[MASKED[SWE]].isel(time=0).values
    nan_frac = float(np.isnan(first).mean())
    cover_max = float(np.nanmax(check[MASKED[SCF]].isel(time=0).values))
    print(
        f"time coord identical; NaN fraction {nan_frac:.4f} vs mask-0 fraction "
        f"{float((~valid).mean()):.4f}; cover max {cover_max:.3f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
