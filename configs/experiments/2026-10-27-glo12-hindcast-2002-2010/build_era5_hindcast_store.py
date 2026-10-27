# mypy: ignore-errors
# ruff: noqa
# ruff: noqa: E501
"""ERA5 4deg daily atmosphere store aligned with the GLORYS12 4deg ocean for the 2002-2010 hindcast (adapted from
build_glo12_era5_store.py): ERA5 stamps T0..T1 relabelled -24 h (day-D mean stamped D 00Z = the GLORYS/GLO12 stamp),
land/sea-surface fractions and 5-day ice (linearly interpolated to daily) from the GLORYS12 4deg store, only the 67 ACE2S
model variables + ak/bk, dims lat/lon. Built variable by variable in parallel subprocesses.

    python build_era5_hindcast_store.py <out.zarr>            # driver
    python build_era5_hindcast_store.py <out.zarr> <var>      # worker
"""

import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

era5_url = "gs://vcm-ml-intermediate/2026-09-08-era5-4deg-8layer-daily-1940-2025.zarr"
ocean_url = "gs://vcm-ml-intermediate/2026-10-21-glorys12-ocean-4deg-19level-5day-1993-2025.zarr"
out = sys.argv[1]
T0, T1 = (
    "2001-11-26T00:00",
    "2011-07-01T00:00",
)  # ERA5 stamps (window end); relabelled -24 h below -> 2001-11-25..2011-06-30
names = (pathlib.Path(__file__).parent / "era5_model_vars.txt").read_text().split()

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
    ds = (
        da.to_dataset(name=v)
        .drop_vars([c for c in da.coords if c != "time"])
        .drop_vars("time")
        .chunk({"time": 73})
    )
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
    "ERA5 daily means relabelled to the START of their day (window end - 24 h), the GLORYS12/GLO12 daily-mean stamp"
)
u = xr.open_dataset(ocean_url, engine="zarr", chunks={"time": 1460}, cache=False)
lf = u["land_fraction"].load()
ssf = u["sea_surface_fraction"].load()
sic5 = u["ocean_sea_ice_fraction"].sel(time=slice("2001-11-01", "2011-08-01")).load()
sic_daily = sic5.interp(
    time=a.time.values, method="linear", kwargs={"fill_value": "extrapolate"}
).clip(0, 1)
sic_daily = sic_daily.where(sic5.isel(time=0).notnull()).load()
print(
    "daily ice ok, NaN days at a pack cell:",
    int(np.isnan(sic_daily.isel(lat=20, lon=40)).sum()),
    flush=True,
)
sif = (sic_daily.fillna(0) * ssf).where(ssf > 0)
of = (1 - lf - sif.fillna(0)).clip(0, 1)
a["land_fraction"] = lf.astype("float32")
a["land_fraction"].attrs = {
    "long_name": "land fraction (GLORYS12 4deg build)",
    "units": "1",
}
a["sea_surface_fraction"] = ssf.astype("float32")
a["sea_surface_fraction"].attrs = {
    "long_name": "ocean fraction of the cell (GLORYS12 4deg build)",
    "units": "1",
}
a["sea_ice_fraction"] = sif.astype("float32").transpose("time", "lat", "lon")
a["sea_ice_fraction"].attrs = {
    "long_name": "sea ice fraction of the whole cell (GLORYS12 concentration x sea_surface_fraction), 5-day values linearly interpolated to daily",
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
    f"{era5_url} (time slice, -24h relabel, lat/lon dims) + fractions and 5-day ice (interpolated) from {ocean_url}"
)
era5_vars = [v for v in keep if "time" in a[v].dims and "lat" in a[v].dims]
small = [
    v for v in keep if "time" in a[v].dims and "lat" not in a[v].dims
]  # global_mean_co2: written eagerly with the skeleton
a.to_zarr(out, mode="w", consolidated=False, compute=False)
a[[v for v in a.data_vars if v not in era5_vars]].to_zarr(
    out, mode="r+", consolidated=False
)
print(
    "skeleton written;",
    len(era5_vars),
    "ERA5 variables to fill; eager:",
    small,
    flush=True,
)


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
chk = xr.open_zarr(out)
print(
    "done",
    dict(chk.sizes),
    "co2 NaN frac",
    float(np.isnan(chk["global_mean_co2"].values).mean()),
    "TMP2m NaN frac",
    float(np.isnan(chk["TMP2m"].isel(time=[0, -1]).values).mean()),
    flush=True,
)
