"""Build the shared static snow-validity mask for the masked-snow arms.

Valid (mask = 1) where land_fraction >= 0.5 AND NOT ice sheet, with ice sheet
defined as any of (mirroring explore2 landatm_utils.icesheet_mask, plus a
max-SWE criterion):

  1. CM4 ``glac_fraction`` >= 0.1 (static; read from the CM4 6-HOURLY store --
     the daily store does not carry it). Strict on purpose, so 1-degree
     ice-margin mixed cells are excluded.
  2. ERA5 local-summer (JJA in NH / DJF in SH) mean snow-cover fraction > 0.5,
     from the ERA5 daily store at a stride.
  3. ERA5 per-cell max SWE > 2000 kg/m2 (2-day-strided full record) -- removes
     deep ice-sheet-margin and cap-pinned cells the other criteria miss.
     Landmarks: seasonal max ~1365, margin ring ~2732, deep margin ~8063.

One mask serves BOTH datasets (identical grids), keeping the masked domains
comparable across ERA5 and CM4 arms. land_fraction is taken from the ERA5
daily store; the two stores' land fractions agree at 1 degree.

Prints the cell count removed by each criterion, writes snow_mask.nc
(variable ``mask``: 1.0 valid / 0.0 masked) and a sanity-map PNG beside it.
"""

import argparse
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ERA5_DAILY = (
    "gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025/"
    "2026-08-07-era5-1deg-8layer-daily-1940-2025.zarr"
)
CM4_6HOURLY = (
    "gs://vcm-ml-intermediate/"
    "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr"
)
GLACIER_THRESHOLD = 0.1
SUMMER_SCF_THRESHOLD = 0.5
MAX_SWE_THRESHOLD = 2000.0
SCF_STRIDE_DAYS = 8
SWE_STRIDE_DAYS = 2


def _gcs_token():
    from google.oauth2.credentials import Credentials

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    ).stdout.strip()
    return Credentials(token=token)


def _lat_name(ds: xr.Dataset) -> str:
    return "latitude" if "latitude" in ds.coords else "lat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="short time sample")
    parser.add_argument("--max-swe-threshold", type=float, default=MAX_SWE_THRESHOLD)
    args = parser.parse_args()

    so = {"token": _gcs_token()}
    era5 = xr.open_zarr(ERA5_DAILY, consolidated=False, storage_options=so)
    cm4_6h = xr.open_zarr(CM4_6HOURLY, consolidated=False, storage_options=so)

    land = era5["land_fraction"].values >= 0.5

    glacier = cm4_6h["glac_fraction"].values >= GLACIER_THRESHOLD

    n = era5.sizes["time"] if not args.dev else 400
    scf = era5["surface_snow_area_fraction"].isel(time=slice(0, n, SCF_STRIDE_DAYS))
    months = scf["time"].dt.month
    jja = scf.isel(time=np.isin(months, [6, 7, 8])).mean("time").values
    djf = scf.isel(time=np.isin(months, [12, 1, 2])).mean("time").values
    lat = era5[_lat_name(era5)].values
    summer_scf = np.where(lat[:, None] >= 0, jja, djf)
    perm_snow = summer_scf > SUMMER_SCF_THRESHOLD

    swe = era5["surface_snow_amount"].isel(time=slice(0, n, SWE_STRIDE_DAYS))
    max_swe = swe.max("time").values
    huge_swe = max_swe > args.max_swe_threshold

    valid = land & ~glacier & ~perm_snow & ~huge_swe

    n_land = int(land.sum())
    print(f"land cells (land_fraction >= 0.5):            {n_land}")
    for label, crit in (
        (f"glac_fraction >= {GLACIER_THRESHOLD}", glacier),
        (f"summer SCF > {SUMMER_SCF_THRESHOLD}", perm_snow),
        (f"max SWE > {args.max_swe_threshold:g}", huge_swe),
    ):
        print(f"  land cells removed by {label:24s}: {int((land & crit).sum())}")
    removed = int((land & (glacier | perm_snow | huge_swe)).sum())
    print(f"  land cells removed by any criterion:          {removed}")
    print(f"valid cells: {int(valid.sum())} ({valid.sum() / land.sum():.1%} of land)")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    mask = xr.Dataset(
        {"mask": (("lat", "lon"), valid.astype(np.float32))},
        coords={
            "lat": era5[_lat_name(era5)].values,
            "lon": era5["longitude" if "longitude" in era5.coords else "lon"].values,
        },
        attrs={
            "description": (
                "Snow-variable validity mask: 1 = valid (land, not ice sheet), "
                "0 = masked. Criteria: land_fraction >= 0.5; excluded if CM4 "
                f"glac_fraction >= {GLACIER_THRESHOLD} or ERA5 local-summer mean "
                f"SCF > {SUMMER_SCF_THRESHOLD} or ERA5 max SWE > "
                f"{args.max_swe_threshold:g} kg/m2."
            )
        },
    )
    mask.to_netcdf(
        f"{out_dir}/snow_mask.nc",
        encoding={"mask": {"dtype": "int8", "zlib": True}},
    )

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    display = np.where(land, np.where(valid, 2.0, 1.0), 0.0)
    ax.pcolormesh(mask["lon"], mask["lat"], display, cmap="viridis")
    ax.set_title("snow mask: 0 non-land (masked), 1 ice sheet (masked), 2 valid")
    fig.savefig(f"{out_dir}/snow_mask.png", dpi=110)
    print(f"wrote {out_dir}/snow_mask.nc and snow_mask.png")


if __name__ == "__main__":
    main()
