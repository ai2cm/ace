#!/usr/bin/env python
# ruff: noqa: E501
"""Build the extra inputs/targets the CM4-line ACE2.5 atmosphere needs to train on the 1deg C96 SHiELD AMIP dataset.

Question (Troy 2026-11-23): fine-tuning the random-CO2 CM4 ACE2.5 on ERA5 always loses its CO2 response; is that ERA5 or
any fine-tune to a different climate? Test = the same fine-tune on C96 SHiELD AMIP (a model with a physical CO2 response).

The SHiELD store (2026-06-08-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/ic_XXXX.zarr) has global_mean_co2
(volume mixing ratio), PRMSL and total_frozen_precipitation_rate, but no lake fraction and no surface wind stresses. This
writes a zarr on one member's time axis with
  carbon_dioxide                                = global_mean_co2 x 1e6               (time,)
  lake_fraction                                 = CM4 piControl 1deg static lake map  (lat, lon)
  eastward/northward_surface_wind_stress        = ZERO PLACEHOLDERS                   (time, lat, lon)
The placeholders exist only because the checkpoint's output list is fixed; the training config gives them loss weight 0,
and their validation/inference metrics are meaningless.

    python build_ace2p5_shield_extra.py --member ic_0001
"""

import argparse
import concurrent.futures

import numpy as np
import xarray as xr
import zarr

SHIELD = "/climate-default/2026-06-08-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset"
p = argparse.ArgumentParser()
p.add_argument("--member", default="ic_0001")
p.add_argument("--shield", default=SHIELD)
p.add_argument(
    "--cm4",
    default="/climate-default/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr",
)
p.add_argument("--out", default="/results/ace2p5_shield_extra.zarr")
p.add_argument("--block", type=int, default=512)
p.add_argument("--threads", type=int, default=8)
a = p.parse_args()

PLACEHOLDERS = ["eastward_surface_wind_stress", "northward_surface_wind_stress"]
src = xr.open_zarr(f"{a.shield}/{a.member}.zarr", chunks=None, consolidated=None)
print("source", dict(src.sizes), src.time.values[[0, -1]], flush=True)
print("variables", sorted(src.data_vars), flush=True)
for v in ["global_mean_co2", "PRMSL", "total_frozen_precipitation_rate", "PRATEsfc"]:
    assert v in src, f"{v} missing from the SHiELD store"
for v in PLACEHOLDERS + ["lake_fraction", "carbon_dioxide"]:
    assert v not in src, f"{v} exists in the SHiELD store: use it, not this store"
tdim = "time"
ydim, xdim = [d for d in src.PRATEsfc.dims if d != tdim]
nt, ny, nx = src.sizes[tdim], src.sizes[ydim], src.sizes[xdim]
# ACE2.5 was trained with PRMSL in hPa and the training config reads this store unscaled: it must be in hPa
prmsl = float(src.PRMSL.isel({tdim: slice(1000, 1008)}).mean())
print("PRMSL units", src.PRMSL.attrs.get("units"), "mean", prmsl, flush=True)
assert 900 < prmsl < 1100, "PRMSL is not in hPa: add overwrite.multiply_scalar"
co2 = src.global_mean_co2.load().values.astype("float64") * 1e6
assert np.isfinite(co2).all() and 250 < co2.min() and co2.max() < 500, (
    co2.min(),
    co2.max(),
)
cm = xr.open_zarr(a.cm4, chunks=None, consolidated=None)
lake = cm.lake_fraction.load().values.astype("float32")
assert lake.shape == (ny, nx), (lake.shape, ny, nx)
assert np.allclose(
    cm[cm.lake_fraction.dims[0]].values, src[ydim].values, atol=1e-3
), "latitude grids differ"

small = xr.Dataset(
    {
        "carbon_dioxide": (
            (tdim,),
            co2.astype("float32"),
            {"units": "ppmv", "long_name": "SHiELD AMIP global_mean_co2 x 1e6"},
        ),
        "lake_fraction": (
            (ydim, xdim),
            lake,
            {
                "units": "fraction",
                "long_name": "lake fraction (CM4 piControl 1deg static map)",
            },
        ),
    },
    coords={tdim: src[tdim].values, ydim: src[ydim].values, xdim: src[xdim].values},
    attrs={
        "note": f"extra inputs/targets for the CM4-line ACE2.5 atmosphere on the C96 SHiELD AMIP {a.member} time axis; the wind stresses are zero placeholders (loss weight 0)"
    },
)
small[tdim].encoding = {
    "units": "hours since 1939-01-01 00:00:00",
    "calendar": src[tdim].encoding.get("calendar", "proleptic_gregorian"),
    "dtype": "float64",
}
small.to_zarr(a.out, mode="w", consolidated=False, zarr_format=2)

g = zarr.open_group(a.out, mode="r+", zarr_format=2)
for dst in PLACEHOLDERS:
    arr = g.create_array(
        dst,
        shape=(nt, ny, nx),
        chunks=(1, ny, nx),
        dtype="float32",
        fill_value=np.nan,
        overwrite=True,
    )
    arr.attrs.update(
        {
            "_ARRAY_DIMENSIONS": [tdim, ydim, xdim],
            "units": "placeholder",
            "long_name": "ZERO PLACEHOLDER, not data",
        }
    )


def do_block(i0: int) -> int:
    i1 = min(i0 + a.block, nt)
    zeros = np.zeros((i1 - i0, ny, nx), "float32")
    for dst in PLACEHOLDERS:
        g[dst][i0:i1] = zeros
    return i1


with concurrent.futures.ThreadPoolExecutor(a.threads) as ex:
    done = 0
    for i1 in ex.map(do_block, range(0, nt, a.block)):
        done = max(done, i1)
        if done % (a.block * 20) < a.block:
            print(f"{done}/{nt} steps written", flush=True)
zarr.consolidate_metadata(a.out)
chk = xr.open_zarr(a.out)
print("written", list(chk.data_vars), dict(chk.sizes))
print(
    "time",
    chk[tdim].values[[0, -1]],
    "matches source",
    bool((chk[tdim].values == src[tdim].values).all()),
)
print(
    "co2 first/last",
    float(chk.carbon_dioxide[0]),
    float(chk.carbon_dioxide[-1]),
    f"lake mean {float(chk.lake_fraction.mean()):.4f}",
)
print(
    "frozen / total precip (source, 8 steps):",
    float(src.total_frozen_precipitation_rate.isel({tdim: slice(1000, 1008)}).mean()),
    float(src.PRATEsfc.isel({tdim: slice(1000, 1008)}).mean()),
)
print(
    "nan check last 3 steps:",
    int(np.isnan(chk.northward_surface_wind_stress[-3:].values).sum()),
)
