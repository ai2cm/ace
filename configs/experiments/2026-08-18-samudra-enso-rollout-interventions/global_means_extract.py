#!/usr/bin/env python3
"""Global-mean monthly reference series for the century-tracking figures."""

import argparse

import cftime
import numpy as np
import xarray as xr
import zarr

VARS = ["sst", "thetao_2", "thetao_6", "thetao_12", "so_2", "so_6", "so_12"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g = zarr.open_group(args.src, mode="r")
    tarr = g["time"]
    times = cftime.num2date(
        tarr[:], units=tarr.attrs["units"], calendar=tarr.attrs["calendar"]
    )
    ym = np.array([t.year * 12 + (t.month - 1) for t in times])
    lat = g["lat"][:].astype("float64")
    w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, g["sst"].shape[2]))
    out = {}
    for v in VARS:
        arr = g[v]
        series = np.zeros(arr.shape[0])
        for s in range(0, arr.shape[0], 360):
            e = min(s + 360, arr.shape[0])
            x = arr[s:e].astype("float64")
            m = np.isfinite(x)
            wa = np.where(m, w2d[None], 0.0)
            series[s:e] = (np.where(m, x, 0.0) * wa).sum(axis=(1, 2)) / wa.sum(
                axis=(1, 2)
            )
        uym = np.unique(ym)
        monthly = np.array([series[ym == k].mean() for k in uym])
        out[v] = monthly
        print(v, "done", flush=True)
    xr.Dataset(
        {v: ("month_index", out[v]) for v in VARS},
        coords={"month_index": ("month_index", np.unique(ym).astype(np.int64))},
    ).to_netcdf(args.out)
    print("done")


if __name__ == "__main__":
    main()
