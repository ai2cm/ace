"""Build a standalone SIDECAR zarr of masked snow channels for one dataset.

The parent stores are never modified. The sidecar holds exactly four
variables, with the time coordinate copied from the parent so the loader's
``merge`` accepts the pair (identical sample times are enforced there):

  surface_snow_amount_masked          parent SWE with NaN where mask == 0
  surface_snow_area_fraction_masked   parent SCF with NaN where mask == 0
  mask_surface_snow_amount_masked         static validity mask (1/0)
  mask_surface_snow_area_fraction_masked  static validity mask (1/0)

NaN in the masked channels is load-bearing: the loss zeroes both prediction
and target where the target is NaN, and the output masker NaN-fills
predictions outside the mask, so store NaNs and mask zeros must agree.

Chunking matches the parent stores: inner time chunk 1, shard 360.

Usage:
  python build_masked_snow_channels.py era5 [--dev]
  python build_masked_snow_channels.py cm4  [--dev]

Writes ./sidecar-out/<parent-name>-snow-masked.zarr; upload with
``gsutil -m rsync -r`` next to the parent store, then copy to weka via
scripts/data_process/gcs_to_weka.sh.
"""

import argparse
import os
import subprocess

import numpy as np
import xarray as xr

SWE = "surface_snow_amount"
SCF = "surface_snow_area_fraction"
PARENTS = {
    "era5": (
        "gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025/"
        "2026-08-07-era5-1deg-8layer-daily-1940-2025.zarr"
    ),
    "cm4": (
        "gs://vcm-ml-intermediate/"
        "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily/"
        "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily.zarr"
    ),
}
SHARD_STEPS = 360


def _gcs_token():
    from google.oauth2.credentials import Credentials

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    ).stdout.strip()
    return Credentials(token=token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(PARENTS))
    parser.add_argument("--dev", action="store_true", help="first 2 shards only")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mask = xr.load_dataset(f"{here}/snow_mask.nc")["mask"].values
    valid = mask > 0.5

    parent_url = PARENTS[args.dataset]
    parent_name = os.path.basename(parent_url).removesuffix(".zarr")
    out = f"{here}/sidecar-out/{parent_name}-snow-masked.zarr"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = xr.open_zarr(
        parent_url, consolidated=False, storage_options={"token": _gcs_token()}
    )
    n = ds.sizes["time"] if not args.dev else 2 * SHARD_STEPS
    dims = ds[SWE].dims
    spatial_coords = {d: ds[d].values for d in dims[1:]}

    static = xr.Dataset(
        {
            f"mask_{SWE}_masked": (dims[1:], mask.astype(np.float32)),
            f"mask_{SCF}_masked": (dims[1:], mask.astype(np.float32)),
        },
        coords=spatial_coords,
    )

    encoding = {
        f"{v}_masked": {
            "chunks": (1, *mask.shape),
            "shards": (SHARD_STEPS, *mask.shape),
        }
        for v in (SWE, SCF)
    }

    first = True
    for start in range(0, n, SHARD_STEPS):
        stop = min(start + SHARD_STEPS, n)
        block = ds[[SWE, SCF]].isel(time=slice(start, stop)).load()
        masked = xr.Dataset(
            {
                f"{SWE}_masked": block[SWE].where(valid),
                f"{SCF}_masked": block[SCF].where(valid),
            }
        )
        if first:
            masked = masked.merge(static)
            masked.to_zarr(out, mode="w", encoding=encoding)
            first = False
        else:
            masked.to_zarr(out, mode="a", append_dim="time")
        print(f"  {args.dataset}: {stop}/{n} steps", flush=True)

    check = xr.open_zarr(out)
    assert check["time"].equals(ds["time"].isel(time=slice(0, n)))
    nan_frac = float(np.isnan(check[f"{SWE}_masked"].isel(time=0).values).mean())
    print(
        f"time coord identical; NaN fraction {nan_frac:.4f} vs mask-0 fraction "
        f"{float((~valid).mean()):.4f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
