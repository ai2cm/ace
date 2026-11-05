# ruff: noqa: E501, UP031
"""Build the CM4-variable-set twin of the 1deg UFS 2026-10-06 5-day store on weka: copy the mirror and append the four
CM4-recipe fields it lacks (hfgeou from the CM4 1deg ocean store, same grid; sfdsi/UI/VI as zeros over ocean), so the 1deg
CM4 checkpoints' variable set loads unchanged. Runs as a CPU Beaker job with weka mounted (launch-build-cm4vars-store.sh).
"""

import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

SRC = "/climate-default/2026-10-06-ufs-replay-ocean-1deg-19level-1994-2023.zarr"
DST = "/climate-default/2026-10-06-ufs-replay-ocean-1deg-19level-1994-2023-cm4vars.zarr"
CM4 = "/climate-default/2026-07-15-om4-picontrol-1deg-ocean-5daily.zarr"

shutil.copytree(SRC, DST, dirs_exist_ok=True)
print("copied", flush=True)
ufs = xr.open_zarr(DST, consolidated=False)
cm4 = xr.open_zarr(CM4, consolidated=False)
assert np.allclose(cm4.lat.values, ufs.lat.values) and np.allclose(
    cm4.lon.values, ufs.lon.values
)
for v in ("hfgeou", "sfdsi", "UI", "VI"):
    if v in ufs:
        print(v, "already present; nothing to do")
        sys.exit(0)
hfgeou = cm4["hfgeou"].load().where(np.isfinite(ufs["deptho"].values))
add = {
    "hfgeou": hfgeou.assign_attrs(
        long_name="Geothermal heat flux (CM4 map, same grid)", units="W m-2", source=CM4
    )
}
zeros = xr.zeros_like(ufs["sea_ice_volume"]).astype("float32")
for v, ln in (
    ("sfdsi", "sea ice salt flux (not in UFS replay; zero)"),
    ("UI", "sea ice x velocity (not in UFS replay; zero)"),
    ("VI", "sea ice y velocity (not in UFS replay; zero)"),
):
    add[v] = zeros.where(np.isfinite(ufs["sea_ice_volume"])).assign_attrs(
        long_name=ln, units=""
    )
xr.Dataset(add).to_zarr(DST, mode="a", consolidated=False)
chk = xr.open_zarr(DST, consolidated=False)
print(
    "added:",
    [v for v in ("hfgeou", "sfdsi", "UI", "VI") if v in chk],
    "| hfgeou ocean mean %.4f" % float(np.nanmean(chk.hfgeou.values)),
)
t = chk.time.values
print(
    "time[0]",
    pd.Timestamp(t[0]),
    "time[-1]",
    pd.Timestamp(t[-1]),
    "n",
    len(t),
    "| vars",
    len(chk.data_vars),
)
