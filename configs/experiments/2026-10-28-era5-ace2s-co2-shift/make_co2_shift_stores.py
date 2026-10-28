"""Write data/era5-4deg-daily-co2-{plus2yr,minus2yr}.zarr for the CO2-clock test.

Each holds global_mean_co2 of the 2026-09-08 ERA5 4deg daily store shifted by +/-2
calendar years (value at t = store CO2 at t+2 yr / t-2 yr), broadcast over the grid,
plus the store's land_fraction; same time axis and encoding. The evaluator merges it
in front of the full store (first source wins). Uploaded as Beaker dataset
era5-4deg-daily-co2-shift-pm2yr (01M62CZ7J9ZXW810YFGD4GH8MN), mounted at /co2shift.
"""

import shutil

import numpy as np
import pandas as pd
import xarray as xr

SRC = "gs://vcm-ml-intermediate/2026-09-08-era5-4deg-8layer-daily-1940-2025.zarr"
KEEP_ENCODING = ("units", "calendar", "dtype", "_FillValue")

ds = xr.open_zarr(SRC)
base = ds[["global_mean_co2", "land_fraction"]].load()
t = pd.DatetimeIndex(base.time.values)
x = t.asi8.astype(float)
v = base.global_mean_co2.values.astype(float)
for name, yrs in [("plus2yr", 2), ("minus2yr", -2)]:
    ts = (t + pd.DateOffset(years=yrs)).asi8.astype(float)
    # flat beyond the store's ends, which lie outside the 1979-2023 run window
    shifted = xr.DataArray(np.interp(ts, x, v), dims="time")
    # fme reads the grid from a (time, lat, lon) variable; the loader broadcasts the
    # store's 1-D original to the same values
    ones = xr.ones_like(base.land_fraction, dtype=float)
    out = base.copy()
    out["global_mean_co2"] = (shifted * ones).transpose("time", "latitude", "longitude")
    out["global_mean_co2"].attrs = dict(
        base.global_mean_co2.attrs,
        notes=base.global_mean_co2.attrs["notes"]
        + f" SHIFTED: value at t is the store value at t{yrs:+d} years.",
    )
    out.attrs = {"history": f"global_mean_co2 of {SRC} shifted by {yrs:+d} years"}
    for c in out.variables:
        enc = ds[c].encoding if c in ds.variables else {}
        out[c].encoding = {k: e for k, e in enc.items() if k in KEEP_ENCODING}
    path = f"data/era5-4deg-daily-co2-{name}.zarr"
    shutil.rmtree(path, ignore_errors=True)
    out.chunk({"time": 1461}).to_zarr(path, zarr_format=2, consolidated=True)
    check = xr.open_zarr(path).global_mean_co2.values[:, 0, 0]
    print(name, {y: round(check[t.year == y].mean() * 1e6, 2) for y in (1979, 2023)})
