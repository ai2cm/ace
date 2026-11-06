"""Surface-fraction store for coupling the ERA5 1deg 6-hourly atmosphere (2026-03-19 store) to the UFS 1deg replay ocean.

Mirrors the CM4 1deg "coupled-atmosphere" store and the 4deg ERA5/UFS build (scratchpad era5_coupled/build_parallel.py):
land_fraction and sea_surface_fraction are the UFS/MOM6 statics; sea_ice_fraction = UFS ice concentration x
sea_surface_fraction (fraction of the whole cell, NaN where there is no ocean, as in the CM4 store); ocean_fraction =
1 - land_fraction - sea_ice_fraction. The UFS store is 5-daily (stamps 18Z every 5 days); the ice is linearly
interpolated to the ERA5 6-hourly stamps; before the first UFS stamp (1994-01-05T18) it is held, and after the last
(2023-10-10T18) it is the 1994-2022 UFS climatology of the calendar day (the coupled inference loader subsets every
merged atmosphere member to start at the ocean's first time and run to the member's END, so this store must end exactly
where the ERA5 store ends, 2025-12-31T18, for the merged members to share one time axis). Dims are named
latitude/longitude like the ERA5 store it is merged with (fme's MergeDatasetConfig takes the FIRST source for a
variable that both hold, so this store goes first in the merge). Because the merged dataset's properties (incl. the
vertical coordinate the dry-air corrector needs) come from the FIRST member, the ERA5 ak_*/bk_* scalars are copied in
too, as the CM4 coupled-atmosphere stores carry them. Built year by year (the VM has 29 GB).
"""
import sys
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import zarr

out = sys.argv[1]
fs = gcsfs.GCSFileSystem()
ufs = xr.open_zarr(fs.get_mapper("gs://vcm-ml-intermediate/ufs-replay-ocean-1deg-19level-5day/2026-10-06-ufs-replay-ocean-1deg-19level-1994-2023.zarr"), consolidated=True)
era = xr.open_zarr(fs.get_mapper("gs://vcm-ml-intermediate/2026-03-19-era5-1deg-8layer-1940-2025.zarr"), consolidated=True)
lat = era.latitude.values
lon = era.longitude.values
assert np.allclose(lat, ufs.lat.values) and np.allclose(lon, ufs.lon.values)
times = pd.date_range("1994-01-01T00:00", "2025-12-31T18:00", freq="6h")
assert times[-1] == pd.Timestamp(era.time.values[-1])
assert np.all(np.isin(times.values, era.time.values))
lf = ufs["land_fraction"].load().values.astype("float32")
ssf = ufs["sea_surface_fraction"].load().values.astype("float32")
assert np.nanmax(np.abs(lf + ssf - 1)) < 1e-6
print("loading 5-day ice", flush=True)
sic = ufs["ocean_sea_ice_fraction"].load().fillna(0.0).astype("float32")  # (2175, 180, 360)
CH = 28
nt = len(times)
akbk = era[[v for v in era.data_vars if v.startswith(("ak_", "bk_"))]].load()
assert len(akbk.data_vars) == 18
skel = xr.Dataset(
    {
        **{v: ((), akbk[v].values, dict(akbk[v].attrs)) for v in akbk.data_vars},
        "land_fraction": (("latitude", "longitude"), lf, {"long_name": "land fraction (UFS/MOM6)", "units": "1"}),
        "sea_surface_fraction": (("latitude", "longitude"), ssf, {"long_name": "ocean fraction of the cell (UFS/MOM6)", "units": "1"}),
        "sea_ice_fraction": (("time", "latitude", "longitude"), da.zeros((nt, 180, 360), dtype="float32", chunks=(CH, 180, 360)), {"long_name": "sea ice fraction of the whole cell (UFS concentration x sea_surface_fraction), 5-daily ice linearly interpolated to 6-hourly", "units": "1"}),
        "ocean_fraction": (("time", "latitude", "longitude"), da.zeros((nt, 180, 360), dtype="float32", chunks=(CH, 180, 360)), {"long_name": "1 - land_fraction - sea_ice_fraction", "units": "1"}),
    },
    coords={"time": times, "latitude": lat, "longitude": lon},
)
skel.attrs["source"] = "fractions from gs://vcm-ml-intermediate/ufs-replay-ocean-1deg-19level-5day/2026-10-06-ufs-replay-ocean-1deg-19level-1994-2023.zarr on the ERA5 2026-03-19 1deg 6-hourly time axis (1994-2023)"
skel.to_zarr(out, mode="w", consolidated=False, compute=False)
skel[["land_fraction", "sea_surface_fraction", *akbk.data_vars]].to_zarr(out, mode="r+", consolidated=False)
print("skeleton written", dict(skel.sizes), flush=True)
years = sorted(set(times.year))
LAST = pd.Timestamp(sic.time.values[-1])
ref2000 = pd.date_range("2000-01-01T00:00", "2000-12-31T18:00", freq="6h")  # leap-year calendar, 1464 slots
clim_sum = np.zeros((len(ref2000), 180, 360), dtype="float64")
clim_cnt = np.zeros(len(ref2000), dtype="int64")


def slot(t):
    return ref2000.get_indexer(pd.DatetimeIndex([pd.Timestamp(2000, ts.month, ts.day, ts.hour) for ts in t]))


for y in years:
    idx = np.where(times.year == y)[0]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    t = times[i0:i1]
    inrec = t <= LAST
    s6v = np.full((len(t), 180, 360), np.nan, dtype="float32")
    if inrec.any():
        tin = t[inrec]
        # source stamps bracketing the block (+1 on each side), the first stamp held before 1994-01-05T18
        src = sic.sel(time=slice(str(tin[0] - pd.Timedelta("6D")), str(tin[-1] + pd.Timedelta("6D"))))
        s6 = src.interp(time=tin, method="linear").bfill("time")
        if np.isnan(s6.values).any():
            s6 = s6.fillna(sic.isel(time=0))
        s6v[inrec] = s6.values.astype("float32")
        if 1994 <= y <= 2022:  # full years -> climatology by calendar day/hour
            sl = slot(tin)
            np.add.at(clim_sum, sl, s6v[inrec].astype("float64"))
            np.add.at(clim_cnt, sl, 1)
    if (~inrec).any():  # after the UFS record: 1994-2022 climatology of the calendar day
        assert clim_cnt.min() > 0, "climatology incomplete"
        clim = (clim_sum / clim_cnt[:, None, None]).astype("float32")
        s6v[~inrec] = clim[slot(t[~inrec])]
    assert not np.isnan(s6v).any()
    sif = s6v * ssf[None]
    sif = np.where(ssf[None] > 0, sif, np.nan).astype("float32")
    of = np.clip(1.0 - lf[None] - np.nan_to_num(sif), 0.0, 1.0).astype("float32")
    blk = xr.Dataset(
        {"sea_ice_fraction": (("time", "latitude", "longitude"), sif), "ocean_fraction": (("time", "latitude", "longitude"), of)},
        coords={"time": t, "latitude": lat, "longitude": lon},
    ).drop_vars(["time", "latitude", "longitude"])
    blk.to_zarr(out, region={"time": slice(i0, i1)}, mode="r+", consolidated=False)
    print("wrote", y, i0, i1, "ice max", float(np.nanmax(sif)), "from UFS" if inrec.all() else ("climatology" if not inrec.any() else "UFS then climatology"), flush=True)
zarr.consolidate_metadata(out)
chk = xr.open_zarr(out, consolidated=True)
for stamp in ["2000-01-04T18:00", "2023-10-10T18:00", "1994-01-05T18:00"]:  # exact UFS stamps
    s = chk.sel(time=stamp).load()
    ref = ufs.sel(time=stamp).load()
    d = (s.sea_ice_fraction.fillna(0) / xr.where(ssf > 0, ssf, 1) - ref.ocean_sea_ice_fraction.fillna(0)).values
    print(stamp, "max |sif/ssf - ufs ice| =", float(np.nanmax(np.abs(d))), "; of+lf+sif range", float(np.nanmin((s.ocean_fraction + s.land_fraction + s.sea_ice_fraction.fillna(0)).values)), float(np.nanmax((s.ocean_fraction + s.land_fraction + s.sea_ice_fraction.fillna(0)).values)), "; ice NaN cells", int(np.isnan(s.sea_ice_fraction.values).sum()), "(ssf==0:", int((ssf == 0).sum()), ")", flush=True)
mid = chk.sel(time="2000-01-07T06:00").sea_ice_fraction.load()
a = chk.sel(time="2000-01-04T18:00").sea_ice_fraction.load(); b = chk.sel(time="2000-01-09T18:00").sea_ice_fraction.load()
print("midpoint check (2.5 d into the interval): max |mid - (a+b)/2| =", float(np.nanmax(np.abs(mid - (a + b) / 2).values)), flush=True)
print("done", flush=True)

c24 = chk.sel(time="2024-03-15T00:00").sea_ice_fraction.load()
print("2024 climatology sample: NH ice area fraction mean", float(np.nanmean(c24.values[c24.latitude.values > 60])), flush=True)
