# ruff: noqa: E501
"""5-day store from the per-year 6-hourly stores of raw_ingest.py, with the same snapshot/window
classification as the 1994-2023 5-day store (scripts/data_process/configs/ufs-replay-ocean-1deg-19level-5day.yaml)
and windows in phase with it (they end 5k days before 1994-01-05T18).  Optionally appends the four
CM4-recipe fields the UFS replay lacks (hfgeou from the CM4 1deg ocean store; sfdsi/UI/VI zero over ocean),
as build_cm4vars_store.py did for the 1994-2023 twin.  Runs on local disk.

    python raw_ingest_5day.py <dir with YYYY.zarr> <out.zarr> [--cm4vars] [--config <5day yaml>]
"""

import argparse
import glob
import os
import sys

import dacite
import numpy as np
import pandas as pd
import xarray as xr
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data_process"))
from time_coarsen import TimeCoarsenConfig, coarsen  # noqa: E402

REFERENCE_WINDOW_END = pd.Timestamp("1994-01-05T18")
CM4_OCEAN = "gs://vcm-ml-intermediate/2026-07-15-om4-picontrol-1deg-ocean-5daily.zarr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("years_dir")
    ap.add_argument("out")
    ap.add_argument("--cm4vars", action="store_true")
    ap.add_argument(
        "--config",
        default=os.path.join(
            HERE,
            "..",
            "..",
            "data_process",
            "configs",
            "ufs-replay-ocean-1deg-19level-5day.yaml",
        ),
    )
    a = ap.parse_args()
    cfg = dacite.from_dict(
        TimeCoarsenConfig, yaml.safe_load(open(a.config))["time_coarsen"]
    )
    paths = sorted(glob.glob(os.path.join(a.years_dir, "[0-9][0-9][0-9][0-9].zarr")))
    parts = [xr.open_zarr(p, chunks={"time": cfg.factor}) for p in paths]
    static = [v for v in parts[0].data_vars if "time" not in parts[0][v].dims]
    for p in parts[1:]:
        for v in static:
            assert np.array_equal(p[v].values, parts[0][v].values, equal_nan=True), v
    ds = xr.concat([p.drop_vars(static) for p in parts], "time")
    ds = xr.merge([ds, parts[0][static]])
    t = pd.DatetimeIndex(ds.time.values)
    assert (
        np.diff(t.values) == np.timedelta64(6, "h")
    ).all(), "6-hourly series has gaps"
    step = pd.Timedelta(hours=6)
    # first timestep of the first full window in phase with the reference store
    window = cfg.factor * step
    k = int(np.ceil((t[0] - (REFERENCE_WINDOW_END + step)) / window))
    start = REFERENCE_WINDOW_END + step + k * window
    ds = ds.sel(time=slice(start, None))
    n = (ds.sizes["time"] // cfg.factor) * cfg.factor
    ds = ds.isel(time=slice(0, n))
    out = coarsen(ds, cfg)
    assert (
        (REFERENCE_WINDOW_END - pd.DatetimeIndex(out.time.values)) % window
        == pd.Timedelta(0)
    ).all()
    print(
        "5-day steps",
        out.sizes["time"],
        str(out.time.values[0])[:13],
        "..",
        str(out.time.values[-1])[:13],
        flush=True,
    )
    if a.cm4vars:
        import gcsfs

        cm4 = xr.open_zarr(
            gcsfs.GCSFileSystem().get_mapper(CM4_OCEAN), consolidated=False
        )
        assert np.allclose(cm4.lat.values, out.lat.values) and np.allclose(
            cm4.lon.values, out.lon.values
        )
        out["hfgeou"] = (
            cm4["hfgeou"]
            .load()
            .where(np.isfinite(out["deptho"].values))
            .reset_coords(drop=True)
            .assign_coords(lat=out.lat, lon=out.lon)
            .assign_attrs(
                long_name="Geothermal heat flux (CM4 map, same grid)",
                units="W m-2",
                source=CM4_OCEAN,
            )
        )
        for v, ln in (
            ("sfdsi", "sea ice salt flux"),
            ("UI", "sea ice x velocity"),
            ("VI", "sea ice y velocity"),
        ):
            out[v] = (
                xr.zeros_like(out["sea_ice_volume"])
                .where(out["sea_ice_volume"].notnull())
                .astype("float32")
                .assign_attrs(long_name=f"{ln} (not in UFS replay; zero)", units="")
            )
    nlat, nlon = out.sizes["lat"], out.sizes["lon"]
    out = out.chunk({"time": 1, "lat": nlat, "lon": nlon})
    enc = {
        v: {"chunks": (1, nlat, nlon)} for v in out.data_vars if "time" in out[v].dims
    }
    for v in out.variables:
        out[v].encoding = {}
    out.to_zarr(a.out, mode="w", zarr_format=3, encoding=enc)
    chk = xr.open_zarr(a.out)
    print("wrote", a.out, dict(chk.sizes), len(chk.data_vars), "vars", flush=True)


if __name__ == "__main__":
    main()
