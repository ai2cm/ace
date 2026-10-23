# ruff: noqa: E501
"""ERA5 4deg daily atmosphere store aligned with the GLO12 4deg ocean (2022-2025; adapted from the UFS build) for coupled training/evaluation, built
variable by variable in parallel subprocesses (one dask graph over 86 variables crawled GCS at ~1 MB/s).

- ERA5 daily samples are labelled at 00Z and are the mean of the preceding day; the UFS 5-day samples are labelled at the
  last 6-hourly sample of their window (18Z). Relabelling ERA5 by -6 h makes both "window end minus 6 h", so the ocean stamp
  1994-01-05T18 coincides with an atmosphere stamp and 5 daily atmosphere steps span one ocean step.
- land_fraction, sea_surface_fraction, sea_ice_fraction and ocean_fraction come from the UFS/MOM6 side (as the CM4
  coupled-atmosphere store does): sea_ice_fraction = ocean ice concentration x sea_surface_fraction (fraction of the whole
  cell, NaN where there is no ocean), ocean_fraction = 1 - land_fraction - sea_ice_fraction. Daily ice = mean of the four
  6-hourly UFS samples of the day, labelled 18Z.
- Only the variables the ERA5 4deg ACE2S atmosphere reads or writes are kept (plus ak/bk), dims renamed to lat/lon.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

era5_url = "gs://vcm-ml-intermediate/2026-09-08-era5-4deg-8layer-daily-1940-2025.zarr"
ufs_url = "gs://vcm-ml-intermediate/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025.zarr"  # GLO12: fractions + 5-day ice
out = sys.argv[1]
T0, T1 = (
    "2022-05-25T00:00",
    "2025-12-31T00:00",
)  # ERA5 stamps (window end); relabelled -24 h below
names = (
    open(
        "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/era5_model_vars.txt"
    )
    .read()
    .split()
)

if len(sys.argv) == 3:  # worker: fill one variable's region
    v = sys.argv[2]
    e = xr.open_dataset(era5_url, engine="zarr", chunks=None, cache=False)
    da = e[v].sel(time=slice(T0, T1)).load()
    da = da.rename(
        {
            k: n
            for k, n in {"latitude": "lat", "longitude": "lon"}.items()
            if k in da.dims
        }
    )
    da = da.assign_coords(time=da.time - pd.Timedelta("24h"))
    da.encoding = {}
    ds = da.to_dataset(name=v).drop_vars(list(da.coords)).chunk({"time": 73})
    ds.to_zarr(
        out, region={"time": slice(0, ds.sizes["time"])}, mode="r+", consolidated=False
    )
    print("filled", v, flush=True)
    sys.exit(0)

e = xr.open_dataset(era5_url, engine="zarr", chunks={"time": 365}, cache=False)
keep = [
    v
    for v in names
    if v in e and v not in ("land_fraction", "ocean_fraction", "sea_ice_fraction")
]
keep += [v for v in e.data_vars if v.startswith(("ak_", "bk_"))]
a = e[keep].sel(time=slice(T0, T1)).rename({"latitude": "lat", "longitude": "lon"})
a = a.assign_coords(time=a.time - pd.Timedelta("24h"))
a.time.attrs["note"] = (
    "ERA5 daily means relabelled to the START of their day (window end - 24 h): the mean of day D is stamped D 00Z, which is how GLO12 stamps its daily means"
)
u = xr.open_dataset(ufs_url, engine="zarr", chunks={"time": 1460}, cache=False)
lf = u["land_fraction"].load()
ssf = u["sea_surface_fraction"].load()
sic5 = u["ocean_sea_ice_fraction"].load()  # 5-day daily means stamped 00Z
sic_daily = sic5.interp(
    time=a.time.values, method="linear", kwargs={"fill_value": "extrapolate"}
).clip(0, 1)

sic_daily = sic_daily.where(sic5.isel(time=0).notnull()).load()
print(
    "daily ice ok, NaN days:",
    int(np.isnan(sic_daily.isel(lat=20, lon=40)).sum()),
    flush=True,
)
sif = (sic_daily.fillna(0) * ssf).where(ssf > 0)
of = (1 - lf - sif.fillna(0)).clip(0, 1)
a["land_fraction"] = lf.astype("float32")
a["land_fraction"].attrs = {
    "long_name": "land fraction (GLO12 4deg build)",
    "units": "1",
}
a["sea_surface_fraction"] = ssf.astype("float32")
a["sea_surface_fraction"].attrs = {
    "long_name": "ocean fraction of the cell (GLO12 4deg build)",
    "units": "1",
}
a["sea_ice_fraction"] = sif.astype("float32").transpose("time", "lat", "lon")
a["sea_ice_fraction"].attrs = {
    "long_name": "sea ice fraction of the whole cell (GLO12 concentration x sea_surface_fraction), 5-day values linearly interpolated to daily",
    "units": "1",
}
a["ocean_fraction"] = of.astype("float32").transpose("time", "lat", "lon")
a["ocean_fraction"].attrs = {
    "long_name": "1 - land_fraction - sea_ice_fraction",
    "units": "1",
}
for v in list(a.variables):
    a[v].encoding = {}
a = a.chunk({"time": 73})
a.attrs["source"] = (
    f"{era5_url} (time slice, -24h relabel, lat/lon dims) + fractions and 5-day ice (interpolated) from {ufs_url}"
)
era5_vars = [v for v in keep if "time" in a[v].dims]
a.to_zarr(
    out, mode="w", consolidated=False, compute=False
)  # skeleton: metadata + coords, data lazy
a[[v for v in a.data_vars if v not in era5_vars]].to_zarr(
    out, mode="r+", consolidated=False
)  # fractions + ak/bk
print("skeleton written;", len(era5_vars), "ERA5 variables to fill", flush=True)


def run(v):
    r = subprocess.run(
        [sys.executable, __file__, out, v], capture_output=True, text=True
    )
    return v, r.returncode, (r.stderr or "")[-400:]


with ThreadPoolExecutor(max_workers=12) as ex:
    for v, rc, err in ex.map(run, era5_vars):
        print("ok" if rc == 0 else f"FAILED {err}", v, flush=True)
import zarr

zarr.consolidate_metadata(out)
print("done", flush=True)
