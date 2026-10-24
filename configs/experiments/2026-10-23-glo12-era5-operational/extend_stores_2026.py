# ruff: noqa: E501
"""Extend the GLO12 ocean store and the GLO12-period ERA5 store through 2026 by relabelling 2025 forward one year, so the
coupled evaluator can run a 12-month forecast from the last analysed state (2025-12-24). During a coupled rollout the model
reads from the stores only its initial condition and the FORCING variables (atmosphere: TOA insolation, CO2, surface
geopotential, land fraction; ocean: static land/sea fractions, bathymetry, geothermal flux); everything else in the 2026 part
is a placeholder and the 2026 'target' metrics are meaningless. CO2 is nudged by its 2024->2025 annual increment.
"""

import pathlib

import numpy as np
import pandas as pd
import xarray as xr

S = pathlib.Path(
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/scratchpad/glo12"
)
g = xr.open_zarr(
    S / "2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025-cm4vars-v2.zarr"
)
e = xr.open_zarr(
    S / "era5" / "2026-10-23-era5-4deg-8layer-daily-glo12-coupled-2022-2025.zarr"
)


def extend(ds, start, shift_days, note):
    tail = ds.sel(time=slice(start, None))
    tail = tail.assign_coords(time=tail.time + pd.Timedelta(days=shift_days))
    tail.attrs["note_2026"] = note
    out = xr.concat([ds, tail], dim="time")
    for v in list(out.variables):
        out[v].encoding = {}
    return out


g2 = extend(
    g,
    "2025-01-01",
    365,
    "2026 = 2025 states relabelled +365 d: placeholders for the forcing loader, NOT data",
)
e2 = extend(
    e,
    "2025-01-01",
    365,
    "2026 = 2025 relabelled +365 d: only insolation/CO2/static fields are used as forcing beyond 2025-12-30",
)
# nudge CO2 in the synthetic year by the last observed annual increment
co2 = e2["global_mean_co2"]
y24 = float(co2.sel(time="2024").mean())
y25 = float(co2.sel(time="2025").mean())
inc = y25 - y24
m = co2.time >= np.datetime64("2026-01-01")
e2["global_mean_co2"] = co2.where(~m, co2 + inc)
print("co2 2024/2025 means", y24, y25, "increment applied to 2026", inc)
g2.chunk({"time": 73}).to_zarr(
    S / "2026-10-22-glo12-ocean-4deg-19level-5day-2022-2026ext-cm4vars-v2.zarr",
    mode="w",
    consolidated=True,
)
e2.chunk({"time": 73}).to_zarr(
    S / "era5" / "2026-10-23-era5-4deg-8layer-daily-glo12-coupled-2022-2026ext.zarr",
    mode="w",
    consolidated=True,
)
print(
    "ocean",
    dict(g2.sizes),
    str(g2.time.values[-1])[:10],
    "| era5",
    dict(e2.sizes),
    str(e2.time.values[-1])[:10],
)
