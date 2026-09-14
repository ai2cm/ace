"""Diff the two UFS replay ocean zarrs (2026-06-03 vs 2026-06-29) and extend
the ERA5-vs-replay stress comparison to the Southern Ocean westerlies.

Context: prior coupled program (June-July) used 06-03; ours uses 06-29. The
implied-advection bias map concentrates in the 40-60S ACC band.
"""

import numpy as np
import xarray as xr

OLD = (
    "gs://vcm-ml-intermediate/"
    "2026-06-03-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr"
)
NEW = (
    "gs://vcm-ml-intermediate/"
    "2026-06-29-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr"
)
opts = {"token": "google_default"}
do = xr.open_zarr(OLD, decode_timedelta=True, storage_options=opts)
dn = xr.open_zarr(NEW, decode_timedelta=True, storage_options=opts)
vo, vn = set(do.data_vars), set(dn.data_vars)
print("only in 06-03:", sorted(vo - vn))
print("only in 06-29:", sorted(vn - vo))
print("time: 06-03", do.time.values[0], "->", do.time.values[-1], len(do.time))
print("time: 06-29", dn.time.values[0], "->", dn.time.values[-1], len(dn.time))
sl = slice("1994-01-01", "2021-12-31")
FIELDS = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "eastward_surface_wind_stress",
    "northward_surface_wind_stress",
    "hfds_total_area",
    "sst",
]
print("\nglobal-ocean area-weighted time-mean, 1994-2021 (06-03 vs 06-29):")
w = np.cos(np.deg2rad(do.lat))
for v in FIELDS:
    if v not in do or v not in dn:
        print(f"  {v:32s}: MISSING in one ({v in do}/{v in dn})")
        continue
    a = do[v].sel(time=sl).mean("time")
    b = dn[v].sel(time=sl).mean("time")
    ma = float(a.weighted(w).mean(("lat", "lon"), skipna=True))
    mb = float(b.weighted(w).mean(("lat", "lon"), skipna=True))
    d = b - a
    rms = float(np.sqrt((d**2).weighted(w).mean(("lat", "lon"), skipna=True)))
    print(f"  {v:32s}: {ma:+.4f} vs {mb:+.4f}  diff {mb - ma:+.4f}  rms {rms:.4f}")

print("\nSouthern Ocean westerlies (40-60S zonal mean eastward stress):")
era5 = xr.open_zarr(
    "/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr",
    decode_timedelta=True,
)
if "latitude" in era5.dims:
    era5 = era5.rename({"latitude": "lat", "longitude": "lon"})
era5 = era5.sortby("lat")
e = era5["eastward_surface_stress"].sel(time=sl).mean("time").load()
r = dn["eastward_surface_wind_stress"].sel(time=sl).mean("time").load()
for la in range(-65, -34, 5):
    ev = float(e.sel(lat=slice(la - 2.5, la + 2.5)).mean(skipna=True))
    rv = float(r.sel(lat=slice(la - 2.5, la + 2.5)).mean(skipna=True))
    print(f"  lat {la:+3d}: era5 {ev:+.4f}  replay {rv:+.4f}  diff {ev - rv:+.4f}")
