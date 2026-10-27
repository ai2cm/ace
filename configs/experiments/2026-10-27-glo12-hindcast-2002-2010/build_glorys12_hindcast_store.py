# mypy: ignore-errors
# ruff: noqa
# ruff: noqa: E501
"""Derived GLORYS12 4deg store for the 2002-2010 clean-window hindcast of the GLO12 coupled SamudrACE checkpoint.

Same layout as build_glo12_store_and_stats.py produced for GLO12 (cm4vars-v2): the time-varying fields of the
GLORYS12 reanalysis 4deg store sliced to 2001-12-05..2011-06-26, with sfdsi/UI/VI and hfds_total_area/tauuo/tauvo/wfo/
unaccounted_heating zero-filled over ocean (NaN on land) so the 88-output checkpoint loads unchanged. No stats are built:
the single-checkpoint evaluator uses the normalization embedded in the checkpoint (GLO12 2022-2024 stats).

    python build_glorys12_hindcast_store.py <out.zarr> [T0 T1]
"""

import sys

import fsspec
import xarray as xr

SRC = "gs://vcm-ml-intermediate/2026-10-21-glorys12-ocean-4deg-19level-5day-1993-2025.zarr"
OUT = sys.argv[1]
T0, T1 = (
    (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("2001-12-05", "2011-06-26")
)

g = (
    xr.open_zarr(fsspec.get_mapper(SRC), consolidated=True)
    .sel(time=slice(T0, T1))
    .load()
)
print(
    "loaded",
    dict(g.sizes),
    len(g.data_vars),
    "vars",
    str(g.time.values[0])[:10],
    str(g.time.values[-1])[:10],
    flush=True,
)
ocean = g["thetao_0"].notnull()
for v in [
    "sfdsi",
    "UI",
    "VI",
    "hfds_total_area",
    "tauuo",
    "tauvo",
    "wfo",
    "unaccounted_heating",
]:
    g[v] = xr.zeros_like(g["thetao_0"]).where(ocean).astype("float32")
    g[v].attrs = {
        "long_name": f"zero placeholder over ocean for the UFS-trained ocean's {v} channel (GLORYS12 real usi/vsi NOT used)",
        "units": "1",
    }
g.attrs["source"] = SRC
g.attrs["note"] = (
    "GLORYS12 4deg 5-day store slice + zero placeholders (cm4vars-v2 layout of the GLO12 derived store) for the 2002-2010 hindcast"
)
for v in list(g.variables):
    g[v].encoding = {}
g = g.chunk({"time": 73})
g.to_zarr(OUT, mode="w", consolidated=True)
print("wrote", OUT, flush=True)
