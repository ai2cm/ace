#!/usr/bin/env python3
"""Derive net radiative fluxes for the net-flux ocean arm.

net_longwave_sfc  = DLWRFsfc - ULWRFsfc   (positive down, into the ocean)
net_shortwave_sfc = DSWRFsfc - USWRFsfc   (positive down)

Combining each radiative pair into its net removes the ULWRFsfc channel -
an invertible readout of the sea surface temperature that supplies it -
from the ocean's inputs entirely. Writes a small zarr next to the coupled
5-day dataset plus a netCDF of training-window normalization statistics.
"""

import argparse

import cftime
import numpy as np
import zarr

PAIRS = {
    "net_longwave_sfc": ("DLWRFsfc", "ULWRFsfc"),
    "net_shortwave_sfc": ("DSWRFsfc", "USWRFsfc"),
}
TIME_CHUNK = 360
TRAIN_START = (256, 1)  # year, month of the training window


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="coupled-ocean 5-day zarr")
    ap.add_argument("--out", required=True, help="output zarr path")
    ap.add_argument("--stats-out", required=True, help="stats netCDF path")
    args = ap.parse_args()

    src = zarr.open_group(args.src, mode="r")
    tarr = src["time"]
    times = cftime.num2date(
        tarr[:], units=tarr.attrs["units"], calendar=tarr.attrs["calendar"]
    )
    in_train = np.array([(t.year, t.month) >= TRAIN_START for t in times], dtype=bool)
    out = zarr.open_group(args.out, mode="w")
    for name in ("lat", "lon"):
        c = src[name]
        oc = out.create_array(
            name, shape=c.shape, dtype=c.dtype, dimension_names=(name,)
        )
        oc[:] = c[:]
        oc.attrs.update(dict(c.attrs))
    tout = out.create_array(
        "time", shape=tarr.shape, dtype=tarr.dtype, dimension_names=("time",)
    )
    tout[:] = tarr[:]
    tout.attrs.update(dict(tarr.attrs))

    stats = {}
    for name, (down, up) in PAIRS.items():
        d, u = src[down], src[up]
        ov = out.create_array(
            name,
            shape=d.shape,
            dtype="float32",
            chunks=(TIME_CHUNK,) + d.shape[1:],
            dimension_names=("time", "lat", "lon"),
        )
        ov.attrs.update(
            {
                "units": "W/m**2",
                "long_name": name.replace("_", " ") + " (positive down)",
            }
        )
        s = s2 = n = 0.0
        for start in range(0, d.shape[0], TIME_CHUNK):
            stop = min(start + TIME_CHUNK, d.shape[0])
            net = d[start:stop].astype("float64") - u[start:stop].astype("float64")
            ov[start:stop] = net.astype("float32")
            sel = net[in_train[start:stop]]
            s += sel.sum()
            s2 += (sel**2).sum()
            n += sel.size
        mean = s / n
        std = float(np.sqrt(s2 / n - mean**2))
        stats[name] = (float(mean), std)
        print(f"{name}: train-window mean {mean:.3f} std {std:.3f}")

    zarr.consolidate_metadata(out.store)

    import xarray as xr

    xr.Dataset({k: ((), v[0]) for k, v in stats.items()}).to_netcdf(
        args.stats_out.replace(".nc", "-centering.nc")
    )
    xr.Dataset({k: ((), v[1]) for k, v in stats.items()}).to_netcdf(
        args.stats_out.replace(".nc", "-scaling.nc")
    )
    print("done")


if __name__ == "__main__":
    main()
