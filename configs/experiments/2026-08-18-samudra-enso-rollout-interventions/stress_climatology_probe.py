"""Quantify the ERA5-vs-UFS-replay wind stress climatology mismatch (1994-2021).

The coupled ocean was trained on replay eastward_surface_wind_stress but is
forced by ACE's ERA5-climatology eastward_surface_stress; equatorial current
biases in coupled mode are the suspected response. Prints tropical-Pacific
means and zonal profiles of the time-mean difference.
"""

import numpy as np
import xarray as xr

era5 = xr.open_zarr(
    "/climate-default/2026-03-19-era5-1deg-8layer-1940-2025.zarr", decode_timedelta=True
)
rep = xr.open_zarr(
    "gs://vcm-ml-intermediate/"
    "2026-06-29-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr",
    decode_timedelta=True,
    storage_options={"token": "google_default"},
)
ren = {}
if "latitude" in era5.dims:
    ren = {"latitude": "lat", "longitude": "lon"}
    era5 = era5.rename(ren)
era5 = era5.sortby("lat")
if float(era5.lon.min()) < 0:
    era5 = era5.assign_coords(lon=(era5.lon % 360)).sortby("lon")
sl = slice("1994-01-01", "2021-12-31")
e = era5["eastward_surface_stress"].sel(time=sl).mean("time").load()
r = rep["eastward_surface_wind_stress"].sel(time=sl).mean("time").load()
e = e.interp_like(r) if e.sizes != r.sizes else e
d = e - r


def box(da, la, lo):
    return float(da.sel(lat=slice(*la), lon=slice(*lo)).mean(skipna=True))


print("time-mean eastward stress (N/m2), 1994-2021:")
for name, la, lo in [
    ("nino4 box (5S-5N,160E-150W)", (-5, 5), (160, 210)),
    ("nino34 box (5S-5N,170W-120W)", (-5, 5), (190, 240)),
    ("eq Pacific (5S-5N,140E-80W)", (-5, 5), (140, 280)),
    ("trades S (20S-5S, Pac)", (-20, -5), (140, 280)),
    ("trades N (5N-20N, Pac)", (5, 20), (140, 280)),
]:
    print(
        f"  {name:32s} era5 {box(e, la, lo):+.4f}  "
        f"replay {box(r, la, lo):+.4f}  diff {box(d, la, lo):+.4f}"
    )
zon = d.sel(lat=slice(-20, 20), lon=slice(140, 280)).mean("lon", skipna=True)
print("zonal-mean diff profile (Pacific, N/m2):")
for lat in range(-20, 21, 5):
    print(f"  lat {lat:+3d}: {float(zon.sel(lat=lat, method='nearest')):+.4f}")
rms = float(
    np.sqrt((d.sel(lat=slice(-20, 20), lon=slice(140, 280)) ** 2).mean(skipna=True))
)
print(f"tropical-Pacific RMS of time-mean diff: {rms:.4f} N/m2")
