# mypy: ignore-errors
# ruff: noqa: E501,E731
"""Add the learned-heating targets to a copy of the 4deg UFS training zarr and append their normalization stats.
U_k = (OHC_k - OHC_{k-1})/dt - [F_rebuild_k * ssf + hfgeou * ssf]   (W/m2 per cell, ocean cells)
unaccounted_heating = trailing 73-step mean of U; unaccounted_heating_5day = U.  Also writes stats entries.
"""

import pathlib
import shutil
import time

import numpy as np
import xarray as xr

SRC = pathlib.Path(
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/ufs4deg_ds_1006/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr"
)
DST = pathlib.Path(
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/ufs_uh/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr"
)
STATS_SRC = pathlib.Path(
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/ufs_uh/stats_src"
)
STATS_DST = pathlib.Path(
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/ufs_uh/stats"
)
RHO = 1035.0
CP = 3992.0
LV = 2.5e6
LF = 334000.0
T0 = 273.15
DT = 5 * 86400.0
if not DST.exists():
    shutil.copytree(SRC, DST)
    print("copied zarr", flush=True)
ds = xr.open_zarr(DST, consolidated=False)
n = ds.sizes["time"]
t0 = time.time()
idep = np.array([float(np.nanmean(ds[f"idepth_{k}"].values)) for k in range(20)])
dz = np.diff(idep)
ssf = ds["sea_surface_fraction"].values
hfgeou = ds["hfgeou"].values
th = [f"thetao_{k}" for k in range(19)]
F = [
    "DLWRFsfc",
    "ULWRFsfc",
    "DSWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "total_frozen_precipitation_rate",
    "sst",
]
U = np.full((n, ssf.shape[0], ssf.shape[1]), np.nan, np.float32)
prev_ohc = None
prev_sst = None
CH = 146
for i0 in range(0, n, CH):
    sl = slice(i0, min(n, i0 + CH))
    a = ds[th + F].isel(time=sl).load()
    T = np.stack([a[v].values for v in th], -1)
    ohc = RHO * CP * np.nansum(np.where(np.isfinite(T), T, 0) * dz, -1)
    ohc[~np.isfinite(T[..., 0])] = np.nan
    sst = a["sst"].values
    for j in range(ohc.shape[0]):
        k = i0 + j
        if k == 0:
            prev_ohc, prev_sst = ohc[j], sst[j]
            continue
        p_ohc = ohc[j - 1] if j > 0 else prev_ohc
        p_sst = sst[j - 1] if j > 0 else prev_sst
        g = lambda v: a[v].values[j]
        base = (
            g("DSWRFsfc")
            - g("USWRFsfc")
            + g("DLWRFsfc")
            - g("ULWRFsfc")
            - g("LHTFLsfc")
            - g("SHTFLsfc")
            - g("total_frozen_precipitation_rate") * LF
        )
        mass = (
            CP
            * (
                g("PRATEsfc")
                + g("total_frozen_precipitation_rate")
                - g("LHTFLsfc") / LV
            )
            * (p_sst - T0)
        )
        rebuild = (base + mass) * ssf + hfgeou * ssf
        U[k] = (ohc[j] - p_ohc) / DT - rebuild
    prev_ohc, prev_sst = ohc[-1], sst[-1]
    print(f"{sl.stop}/{n} {time.time()-t0:.0f}s", flush=True)
ocean = np.isfinite(ssf) & (ssf > 0) & np.isfinite(U[1])
U[0] = U[
    1
]  # no previous step for the first time; copy (first window is dropped from training anyway)
# trailing 73-step mean (causal), nan-aware, with shorter windows at the start
cs = np.nancumsum(np.where(np.isfinite(U), U, 0), axis=0)
cn = np.cumsum(np.isfinite(U), axis=0)
Us = np.full_like(U, np.nan)
for k in range(n):
    k0 = max(0, k - 72)
    s = cs[k] - (cs[k0 - 1] if k0 > 0 else 0)
    c = cn[k] - (cn[k0 - 1] if k0 > 0 else 0)
    Us[k] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
U = np.where(ocean[None], U, np.nan).astype(np.float32)
Us = np.where(ocean[None], Us, np.nan).astype(np.float32)
lat = ds.lat.values
w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(ssf)
gm = np.nansum(Us * w, axis=(1, 2)) / np.nansum(w * ocean)
print(
    f"unaccounted_heating ocean-area mean: 1994-2023 {np.nanmean(gm):+.3f} W/m2; annual (every 5 yr): {np.round(gm[72::365], 2)}",
    flush=True,
)
out = xr.Dataset(
    {
        "unaccounted_heating": (("time", "lat", "lon"), Us),
        "unaccounted_heating_5day": (("time", "lat", "lon"), U),
    },
    coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
)
out["unaccounted_heating"].attrs = {
    "long_name": "column heating unaccounted for by the rebuilt surface flux, trailing 1-yr mean",
    "units": "W m-2",
    "definition": "trailing 73-step mean of (dOHC/dt - (F_rebuild + hfgeou)*sea_surface_fraction), per cell area, ocean cells",
}
out["unaccounted_heating_5day"].attrs = {
    "long_name": "column heating unaccounted for by the rebuilt surface flux, per 5-day step",
    "units": "W m-2",
}
enc = {
    v: {"chunks": ds["sst"].encoding.get("chunks", (1, lat.size, ds.lon.size))}
    for v in out.data_vars
}
out.chunk({"time": enc["unaccounted_heating"]["chunks"][0]}).to_zarr(
    DST, mode="a", consolidated=False
)
print("fields written", flush=True)
# stats: append entries to copies of the 2026-10-06 stats files
import glob

STATS_DST.mkdir(exist_ok=True, parents=True)
for f in glob.glob(str(STATS_SRC / "*.nc")):
    d = xr.open_dataset(f).load()
    name = pathlib.Path(f).name
    for v, arr in (("unaccounted_heating", Us), ("unaccounted_heating_5day", U)):
        if name == "centering.nc":
            val = float(np.nanmean(arr))
        elif name.startswith("scaling-full-field"):
            val = float(np.nanstd(arr))
        elif name.startswith("scaling-residual"):
            val = float(np.nanstd(arr[1:] - arr[:-1]))
        elif name.startswith("time-mean"):
            d[v] = (("lat", "lon"), np.nanmean(arr, 0))
            continue
        else:
            continue
        d[v] = xr.DataArray(np.float32(val))
    d.to_netcdf(STATS_DST / name)
    print(
        "stats",
        name,
        {
            v: float(d[v])
            for v in ("unaccounted_heating", "unaccounted_heating_5day")
            if d[v].ndim == 0
        },
        flush=True,
    )
print("done", flush=True)
