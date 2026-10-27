# mypy: ignore-errors
# ruff: noqa
"""Make the derived GLORYS12 store compatible with the GLO12-trained checkpoint's masks: replace mask_k by the GLO12 masks and,
wherever the GLO12 mask is ocean but GLORYS12 is NaN (5 column cells at levels 14/16), fill every 3-D variable from the
nearest finite level above in the same column (constant in time). Cells that are ocean in GLORYS but land in GLO12 keep
their data (the checkpoint masks them).
"""

import shutil

import numpy as np
import xarray as xr

P = "2026-10-27-glorys12-ocean-4deg-19level-5day-2001-2011-cm4vars.zarr"
o = xr.open_zarr(P).load()
g = xr.open_zarr(
    "gs://vcm-ml-intermediate/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025.zarr",
    consolidated=None,
    chunks=None,
)
masks = [v for v in o.data_vars if v.startswith("mask_")]
print("mask vars", masks)
nfill = 0
for k in range(19):
    gm = g[f"mask_{k}"].values > 0
    for base in ["thetao", "so", "uo", "vo"]:
        da = o[f"{base}_{k}"]
        bad = gm & np.isnan(da.isel(time=0).values)
        if bad.any():
            jj, ii = np.where(bad)
            for j, i in zip(jj, ii):
                for kk in range(k - 1, -1, -1):
                    src = o[f"{base}_{kk}"].values[:, j, i]
                    if np.isfinite(src).all():
                        da.values[:, j, i] = src
                        nfill += 1
                        break
                else:
                    raise RuntimeError(
                        f"no finite level above for {base} level {k} cell {j},{i}"
                    )
    o[f"mask_{k}"] = g[f"mask_{k}"].load().astype(o[f"mask_{k}"].dtype)
for v in masks:
    if v not in [f"mask_{k}" for k in range(19)]:
        o[v] = g[v].load().astype(o[v].dtype)
        print("replaced", v)
o.attrs["mask_note"] = (
    "mask_k replaced by the GLO12 4deg masks (the checkpoint's); 3-D fields filled from the level above where GLO12 mask is ocean and GLORYS12 is NaN"
)
print("filled column values:", nfill)
# verify: no NaN where GLO12 mask is ocean, any level, any 3-D var, first/last time
for k in range(19):
    gm = g[f"mask_{k}"].values > 0
    for base in ["thetao", "so", "uo", "vo"]:
        x = o[f"{base}_{k}"].isel(time=[0, -1]).values
        assert not (np.isnan(x) & gm[None]).any(), (base, k)
for v in list(o.variables):
    o[v].encoding = {}
tmp = P + ".new"
shutil.rmtree(tmp, ignore_errors=True)
o.chunk({"time": 73}).to_zarr(tmp, mode="w", consolidated=True)
shutil.rmtree(P)
shutil.move(tmp, P)
print("rewrote", P)
