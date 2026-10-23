# ruff: noqa: E501
"""Derived GLO12 4deg training store for the UFS-trained SamudrACE ocean + its normalization stats.

- Source: gs://vcm-ml-intermediate/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025.zarr (261 five-day daily-mean
  states, 2022-06-03..2025-12-24, stamps 00Z).
- Adds/overwrites the three placeholder channels the UFS-trained ocean reads and writes (sfdsi, UI, VI) as zeros over ocean, NaN
  over land, exactly as the cm4vars-uh UFS training store carries them (zero over ocean, NaN on land).
- hfds_total_area / tauuo / tauvo / wfo stay all-NaN (dropped from out_names in the fine-tune); unaccounted_heating is not added.
- Stats: GLO12 2022-06-01..2024-12-31 (the training years; 2025 untouched), same formulas as scripts/data_process/get_stats.py
  (centering = mean over time/lat/lon, scaling-full-field = std, scaling-residual = std of the 5-day difference,
  time-mean = mean over time); placeholder + unaccounted_heating entries copied from the UFS stats bundle
  01M5D5GG2FB3MWWDD5GGSH8QR2 so the checkpoint's normalization keeps the values it was trained with.
"""

import pathlib

import fsspec
import numpy as np
import xarray as xr

S = pathlib.Path(
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/glo12"
)
SRC = "gs://vcm-ml-intermediate/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025.zarr"
OUT = S / "2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025-cm4vars.zarr"
STATS = S / "stats"
STATS.mkdir(exist_ok=True)

g = xr.open_zarr(fsspec.get_mapper(SRC), consolidated=True).load()
print("loaded", dict(g.sizes), len(g.data_vars), "vars", flush=True)
ocean = g["thetao_0"].notnull()  # (time, lat, lon) ocean mask incl. time-constant land
for v in [
    "sfdsi",
    "UI",
    "VI",
]:  # GLO12 carries real UI/VI (usi/vsi); the UFS-trained ocean only ever saw zeros, so they are overwritten
    g[v] = xr.zeros_like(g["thetao_0"]).where(ocean).astype("float32")
g["sfdsi"].attrs = {
    "long_name": "placeholder (zero over ocean) for the UFS-trained ocean's sfdsi channel",
    "units": "kg m-2 s-1",
}
g["UI"].attrs = {
    "long_name": "placeholder (zero over ocean) for the UFS-trained ocean's UI channel; GLO12's real usi is NOT used",
    "units": "m s-1",
}
g["VI"].attrs = {
    "long_name": "placeholder (zero over ocean) for the UFS-trained ocean's VI channel; GLO12's real vsi is NOT used",
    "units": "m s-1",
}
g.attrs["source"] = SRC
g.attrs["note"] = (
    "GLO12 4deg 5-day store + zero placeholders sfdsi/UI/VI for the UFS-trained SamudrACE ocean (cm4vars layout)"
)
for v in list(g.variables):
    g[v].encoding = {}
g = g.chunk({"time": 73} if "time" in g.dims else {})
g.to_zarr(OUT, mode="w", consolidated=True)
print("wrote", OUT, flush=True)

# ---- stats on the training years
tr = g.sel(time=slice("2022-06-01", "2024-12-31"))
dims = ["time", "lat", "lon"]
tvars = [v for v in tr.data_vars if "time" in tr[v].dims]
svars = [v for v in tr.data_vars if "time" not in tr[v].dims]
centering = xr.merge([tr[tvars].mean(dim=dims), tr[svars].mean(dim=["lat", "lon"])])
scaling_ff = xr.merge([tr[tvars].std(dim=dims), tr[svars].std(dim=["lat", "lon"])])
scaling_res = xr.merge(
    [tr[tvars].diff("time").std(dim=dims), tr[svars].std(dim=["lat", "lon"])]
)
time_mean = xr.merge([tr[tvars].mean(dim="time"), tr[svars]])
ufs = {
    n: xr.open_dataset(S / "ufs_stats" / f"{n}.nc")
    for n in ["centering", "scaling-full-field", "scaling-residual"]
}
ufs_tm = xr.open_dataset(S / "ufs_stats" / "time-mean.nc")
copy = ["sfdsi", "UI", "VI", "unaccounted_heating"]
for name, ds, src in [
    ("centering", centering, ufs["centering"]),
    ("scaling-full-field", scaling_ff, ufs["scaling-full-field"]),
    ("scaling-residual", scaling_res, ufs["scaling-residual"]),
]:
    for v in copy:
        ds[v] = src[v]
    ds.attrs["history"] = (
        f"GLO12 stats 2022-06-01..2024-12-31 from {OUT.name}; {copy} copied from UFS stats 01M5D5GG2FB3MWWDD5GGSH8QR2"
    )
    bad = [
        v
        for v in ds.data_vars
        if not np.isfinite(ds[v].values).all()
        and v not in ("hfds_total_area", "tauuo", "tauvo", "wfo")
    ]
    print(
        name,
        "n",
        len(ds.data_vars),
        "non-finite (expected only the 4 NaN diagnostics):",
        bad,
        flush=True,
    )
    ds.to_netcdf(STATS / f"{name}.nc")
for v in copy:
    if v in ufs_tm:
        time_mean[v] = ufs_tm[v]
time_mean.to_netcdf(STATS / "time-mean.nc")
print("stats written to", STATS, flush=True)
