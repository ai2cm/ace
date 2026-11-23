#!/usr/bin/env python
# ruff: noqa: E501
"""Build the extra inputs/targets the CM4-line ACE2.5 atmosphere needs to train on the 1deg C96 SHiELD AMIP dataset.

Question (Troy 2026-11-23): fine-tuning the random-CO2 CM4 ACE2.5 on ERA5 always loses its CO2 response; is that ERA5 or
any fine-tune to a different climate? Test = the same fine-tune on C96 SHiELD AMIP (a model with a physical CO2 response).

The SHiELD store (2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_XXXX, monthly files) has
global_mean_co2 (volume mixing ratio), no lake fraction, frozen precipitation as three species, and neither PRMSL nor the
surface wind stresses. This writes a zarr on one member's time axis with
  carbon_dioxide                   = global_mean_co2 x 1e6                 (time,)
  lake_fraction                    = CM4 piControl 1deg static lake map    (lat, lon)
  total_frozen_precipitation_rate  = GRAUPELsfc + ICEsfc + SNOWsfc         (time, lat, lon)
  PRMSL, eastward/northward_surface_wind_stress = ZERO PLACEHOLDERS        (time, lat, lon)
The placeholders exist only because the checkpoint's output list is fixed; the training config gives them loss weight 0,
and their validation/inference metrics are meaningless.

    python build_ace2p5_shield_extra.py --member ic_0001
"""

import argparse
import concurrent.futures
import glob

import numpy as np
import xarray as xr
import zarr

SHIELD = "/climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs"
p = argparse.ArgumentParser()
p.add_argument("--member", default="ic_0001")
p.add_argument("--shield", default=SHIELD)
p.add_argument(
    "--cm4",
    default="/climate-default/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr",
)
p.add_argument("--out", default="/results/ace2p5_shield_extra.zarr")
p.add_argument("--procs", type=int, default=12)
a = p.parse_args()

files = sorted(glob.glob(f"{a.shield}/{a.member}/*.nc"))
assert files, f"no netCDF files under {a.shield}/{a.member}"
FROZEN = ["GRAUPELsfc", "ICEsfc", "SNOWsfc"]
PLACEHOLDERS = [
    "PRMSL",
    "eastward_surface_wind_stress",
    "northward_surface_wind_stress",
]
first = xr.open_dataset(files[0], decode_times=False)
print("first file", files[0], dict(first.sizes), flush=True)
print("variables", sorted(first.data_vars), flush=True)
for v in FROZEN + ["global_mean_co2", "PRATEsfc"]:
    assert v in first, f"{v} missing from the SHiELD files"
for v in PLACEHOLDERS + ["total_frozen_precipitation_rate", "lake_fraction"]:
    assert v not in first, f"{v} exists in the SHiELD files: use it, not this store"
tdim = "time"
ydim, xdim = [d for d in first.PRATEsfc.dims if d != tdim]
ny, nx = first.sizes[ydim], first.sizes[xdim]
time_attrs = dict(first[tdim].attrs)
print("time attrs", time_attrs, "horizontal dims", ydim, xdim, flush=True)


def read_meta(f: str):
    d = xr.open_dataset(f, decode_times=False)
    assert d[tdim].attrs.get("units") == time_attrs.get("units"), f
    return d[tdim].values, d.global_mean_co2.values


def read_frozen(f: str):
    d = xr.open_dataset(f, decode_times=False)
    return sum(d[v].values.astype("float64") for v in FROZEN).astype("float32")


with concurrent.futures.ProcessPoolExecutor(a.procs) as ex:
    meta = list(ex.map(read_meta, files))
lens = [len(m[0]) for m in meta]
offs = np.concatenate([[0], np.cumsum(lens)])
nt = int(offs[-1])
tvals = np.concatenate([m[0] for m in meta])
assert np.all(np.diff(tvals) > 0), "time axis is not increasing across files"
co2 = np.concatenate([m[1] for m in meta]).astype("float64") * 1e6
assert np.isfinite(co2).all() and 250 < co2.min() and co2.max() < 500, (
    co2.min(),
    co2.max(),
)
cm = xr.open_zarr(a.cm4, chunks=None, consolidated=None)
lake = cm.lake_fraction.load().values.astype("float32")
assert lake.shape == (ny, nx), (lake.shape, ny, nx)
cmlat = cm[cm.lake_fraction.dims[0]].values
assert np.allclose(cmlat, first[ydim].values, atol=1e-3), "latitude grids differ"

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
    coords={
        tdim: (tdim, tvals, time_attrs),
        ydim: first[ydim].values,
        xdim: first[xdim].values,
    },
    attrs={
        "note": f"extra inputs/targets for the CM4-line ACE2.5 atmosphere on the C96 SHiELD AMIP {a.member} time axis; PRMSL and the wind stresses are zero placeholders (loss weight 0)"
    },
)
small.to_zarr(a.out, mode="w", consolidated=False, zarr_format=2)

g = zarr.open_group(a.out, mode="r+", zarr_format=2)
LONG = {
    "total_frozen_precipitation_rate": (
        "kg/m**2/s",
        "GRAUPELsfc + ICEsfc + SNOWsfc",
    ),
    **{v: ("placeholder", "ZERO PLACEHOLDER, not data") for v in PLACEHOLDERS},
}
for dst, (units, long_name) in LONG.items():
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
            "units": units,
            "long_name": long_name,
        }
    )

with concurrent.futures.ProcessPoolExecutor(a.procs) as ex:
    for k, froz in enumerate(ex.map(read_frozen, files)):
        i0, i1 = int(offs[k]), int(offs[k + 1])
        assert froz.shape == (i1 - i0, ny, nx) and np.isfinite(froz).all(), files[k]
        g["total_frozen_precipitation_rate"][i0:i1] = froz
        zeros = np.zeros_like(froz)
        for v in PLACEHOLDERS:
            g[v][i0:i1] = zeros
        if k % 50 == 0:
            print(f"{k}/{len(files)} files, {i1}/{nt} steps written", flush=True)
zarr.consolidate_metadata(a.out)
chk = xr.open_zarr(a.out)
print("written", list(chk.data_vars), dict(chk.sizes))
print("time", chk[tdim].values[[0, -1]])
print(
    "co2 first/last",
    float(chk.carbon_dioxide[0]),
    float(chk.carbon_dioxide[-1]),
    f"lake mean {float(chk.lake_fraction.mean()):.4f}",
)
last = xr.open_dataset(files[-1])
print(
    "frozen / total precip, last file:",
    float(chk.total_frozen_precipitation_rate[-lens[-1] :].mean()),
    float(last.PRATEsfc.mean()),
)
print(
    "nan check last 3 steps:",
    int(np.isnan(chk.total_frozen_precipitation_rate[-3:].values).sum()),
    int(np.isnan(chk.PRMSL[-3:].values).sum()),
)
