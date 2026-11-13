#!/usr/bin/env python
# ruff: noqa: E501
"""Build the extra ERA5-axis inputs the CM4-line ACE2.5 atmosphere needs to train on the 2026-03-19 ERA5 1deg 8-layer store.

ACE2.5 (Spencer's CM4 random-CO2 atmosphere) reads carbon_dioxide in ppmv and a lake_fraction, and writes the surface wind
stress as eastward/northward_surface_wind_stress in CM4's sign convention; the ERA5 store has global_mean_co2 (volume mixing
ratio), no lake fraction, and eastward/northward_surface_stress in the opposite sign (the coupled configs apply -1 between
the two, make_configs.py atmosphere_output_scale). This writes a zarr on the ERA5 time axis with
  carbon_dioxide                 = global_mean_co2 x 1e6                       (time,)
  lake_fraction                  = CM4 piControl 1deg static lake map          (latitude, longitude)
  eastward_surface_wind_stress   = -eastward_surface_stress                    (time, latitude, longitude)
  northward_surface_wind_stress  = -northward_surface_stress
chunked like the source (1 step per chunk), to be merged with the ERA5 store in the training config.

    python build_ace2p5_era5_extra.py [--era5 PATH] [--cm4 PATH] [--out PATH]
"""

import argparse
import concurrent.futures

import numpy as np
import xarray as xr
import zarr

p = argparse.ArgumentParser()
p.add_argument(
    "--era5", default="/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr"
)
p.add_argument(
    "--cm4",
    default="/climate-default/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr",
)
p.add_argument("--out", default="/results/ace2p5_era5_extra.zarr")
p.add_argument("--block", type=int, default=512)
p.add_argument("--threads", type=int, default=8)
a = p.parse_args()

era = xr.open_zarr(
    a.era5, chunks=None, consolidated=None
)  # no dask in the Beaker image: lazy indexing + explicit blocks
cm = xr.open_zarr(a.cm4, chunks=None, consolidated=None)
co2 = era.global_mean_co2.load().values * 1e6
fin = np.isfinite(co2)
co2[~fin] = np.interp(np.flatnonzero(~fin), np.flatnonzero(fin), co2[fin])
lake = cm.lake_fraction.load().values.astype("float32")
nt, ny, nx = era.sizes["time"], era.sizes["latitude"], era.sizes["longitude"]
assert lake.shape == (ny, nx)
small = xr.Dataset(
    {
        "carbon_dioxide": (
            ("time",),
            co2.astype("float32"),
            {
                "units": "ppmv",
                "long_name": "global mean CO2 (ERA5 store global_mean_co2 x 1e6)",
            },
        ),
        "lake_fraction": (
            ("latitude", "longitude"),
            lake,
            {
                "units": "fraction",
                "long_name": "lake fraction (CM4 piControl 1deg atmosphere store static map)",
            },
        ),
    },
    coords={
        "time": era.time.values,
        "latitude": era.latitude.values,
        "longitude": era.longitude.values,
    },
    attrs={
        "note": "extra inputs/targets for the CM4-line ACE2.5 atmosphere on the 2026-03-19 ERA5 1deg 8-layer store's time axis"
    },
)
small.time.encoding = {
    "units": "hours since 1940-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "float64",
}
small.to_zarr(a.out, mode="w", consolidated=False, zarr_format=2)

# The two big fields: minus the ERA5 stresses, one timestep per chunk like the source, written block by block in threads.
g = zarr.open_group(a.out, mode="r+", zarr_format=2)
NAMES = {
    "eastward_surface_wind_stress": "eastward_surface_stress",
    "northward_surface_wind_stress": "northward_surface_stress",
}
for dst, srcname in NAMES.items():
    arr = g.create_array(
        dst,
        shape=(nt, ny, nx),
        chunks=(1, ny, nx),
        dtype="float32",
        fill_value=np.nan,
        overwrite=True,
    )
    arr.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time", "latitude", "longitude"],
            "units": "Pa",
            "long_name": f"minus the ERA5 {srcname} (CM4 sign convention)",
        }
    )


def do_block(i0: int) -> int:
    i1 = min(i0 + a.block, nt)
    for dst, srcname in NAMES.items():
        g[dst][i0:i1] = -era[srcname].isel(time=slice(i0, i1)).values.astype("float32")
    return i1


with concurrent.futures.ThreadPoolExecutor(a.threads) as ex:
    done = 0
    for i1 in ex.map(do_block, range(0, nt, a.block)):
        done = max(done, i1)
        if done % (a.block * 20) < a.block:
            print(f"{done}/{nt} steps written", flush=True)
zarr.consolidate_metadata(a.out)
chk = xr.open_zarr(a.out)
print("written", list(chk.data_vars), dict(chk.sizes))
print(
    "co2 1940/2025",
    float(chk.carbon_dioxide[0]),
    float(chk.carbon_dioxide[-1]),
    f"lake mean {float(chk.lake_fraction.mean()):.4f}",
)
print(
    "stress check (should be minus the ERA5 value):",
    float(chk.eastward_surface_wind_stress[min(1000, nt - 1), 90, 180]),
    float(era.eastward_surface_stress[min(1000, nt - 1), 90, 180]),
)
print(
    "nan check last block:",
    int(np.isnan(chk.northward_surface_wind_stress[-3:].values).sum()),
)
