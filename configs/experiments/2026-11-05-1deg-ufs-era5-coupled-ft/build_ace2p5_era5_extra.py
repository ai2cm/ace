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

import dask
import numpy as np
import xarray as xr

p = argparse.ArgumentParser()
p.add_argument(
    "--era5", default="/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr"
)
p.add_argument(
    "--cm4",
    default="/climate-default/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr",
)
p.add_argument("--out", default="/results/ace2p5_era5_extra.zarr")
a = p.parse_args()

era = xr.open_zarr(a.era5, chunks={"time": 256}, consolidated=None)
cm = xr.open_zarr(a.cm4, chunks=None, consolidated=None)
co2 = era.global_mean_co2.load().values * 1e6
fin = np.isfinite(co2)
co2[~fin] = np.interp(np.flatnonzero(~fin), np.flatnonzero(fin), co2[fin])
lake = cm.lake_fraction.load().values.astype("float32")
assert lake.shape == (era.sizes["latitude"], era.sizes["longitude"])
out = xr.Dataset(
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
        "eastward_surface_wind_stress": (-era.eastward_surface_stress)
        .astype("float32")
        .assign_attrs(
            units="Pa",
            long_name="minus the ERA5 eastward_surface_stress (CM4 sign convention)",
        ),
        "northward_surface_wind_stress": (-era.northward_surface_stress)
        .astype("float32")
        .assign_attrs(
            units="Pa",
            long_name="minus the ERA5 northward_surface_stress (CM4 sign convention)",
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
out.time.encoding = {
    "units": "hours since 1940-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "float64",
}
enc = {
    v: {"chunks": (1, 180, 360)}
    for v in ["eastward_surface_wind_stress", "northward_surface_wind_stress"]
}
with dask.config.set(scheduler="threads", num_workers=16):
    out.to_zarr(a.out, mode="w", consolidated=True, encoding=enc)
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
    float(chk.eastward_surface_wind_stress[1000, 90, 180]),
    float(era.eastward_surface_stress[1000, 90, 180]),
)
