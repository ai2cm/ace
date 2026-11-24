# ruff: noqa: E501
"""Ingest UFS replay periods that NOAA's analysis-ready zarrs do not cover (1980-1993, 2024+)
from the raw per-cycle files on s3://noaa-ufs-gefsv13replay-pds, producing the same 6-hourly
output as ufs-replay-pipeline.py.

Each raw MOM6 file is first put on the 0.25-degree Gaussian grid exactly as NOAA's ufs2arco did
for mom6.zarr (see tripolar.py; verified to float precision against the zarr on 1994 files), then
handed unchanged to the pipeline's own per-chunk functions.  The invariant fields (masks,
fractions, depths) and nearest-neighbour fill indices are computed from the zarrs' first
1994 timestep, as in the production run, so the output shares the 1994-2023 store's mask.

    python raw_ingest.py setup  <work_dir>                      # static.pkl + regrid weights
    python raw_ingest.py run    <work_dir> <out.zarr> <start> <end> [--workers N]
"""

import argparse
import datetime
import importlib.util
import json
import logging
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    "ufs_replay_pipeline", os.path.join(HERE, "ufs-replay-pipeline.py")
)
assert _spec is not None and _spec.loader is not None
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
import tripolar  # noqa: E402

BUCKET = "noaa-ufs-gefsv13replay-pds"
HGRID = "noaa-nws-global-pds/fix/mom6/20240416/025/ocean_hgrid.nc"
TEMPLATE_TIME = datetime.datetime(1994, 1, 1, 0)
SCALARS_3D = ["temp", "so"]
SCALARS_2D = ["SSH"] + P.WFO_COMPONENTS + P.HFDS_COMPONENTS
ATMO_VARS = list(P.ATMO_FORCING_VARS) + list(P.ICE_VARS) + P.FROZEN_PRECIP_ACCUM_VARS
SIX_HOURS = datetime.timedelta(hours=6)


def _s3():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def ocean_key(t: datetime.datetime) -> str:
    """The copy NOAA's zarr holds: the first of the two ocean files of the NEXT cycle."""
    c = t + SIX_HOURS
    return f"{BUCKET}/{c:%Y/%m/%Y%m%d%H}/ocn_{t:%Y_%m_%d_%H}.nc"


def atmo_key(t: datetime.datetime) -> str:
    """FV3 surface file valid at ``t`` (a multiple of 3 h).  See FV3_FHR below."""
    fhr = FV3_FHR[t.hour % 6]
    c = t + datetime.timedelta(hours=6 - fhr)
    return f"{BUCKET}/{c:%Y/%m/%Y%m%d%H}/bfg_{c:%Y%m%d%H}_fhr{fhr:02d}_control"


# Forecast hour of the bfg file used for a valid time at (hour % 6).  Cycle directory
# YYYYMMDDHH holds fhr00..09 valid at HH-6..HH+3; fv3.zarr holds fhr00 and fhr03 of each
# cycle (checked bitwise on 1994-06-01), not the fhr06/fhr09 copies of the previous one.
FV3_FHR = {0: 0, 3: 3}


def _fetch(fs, key, local):
    for attempt in range(5):
        try:
            fs.get(key, local)
            return
        except FileNotFoundError:
            raise
        except Exception as e:  # transient S3 errors
            logging.warning("fetch %s failed (%s), retry %d", key, e, attempt)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not fetch {key}")


def setup(work_dir: str, output_grid: str):
    os.makedirs(work_dir, exist_ok=True)
    fs = _s3()
    hgrid = os.path.join(work_dir, "ocean_hgrid.nc")
    if not os.path.exists(hgrid):
        _fetch(fs, HGRID, hgrid)
    ds_ocean = P._clean_ocean_dataset(
        P.open_ocean(P.OCEAN_LOAD_VARS, TEMPLATE_TIME, TEMPLATE_TIME)
    )
    ds_atmo = P.open_atmo(ATMO_VARS)
    ds_ocean, ds_atmo = P._align_times(ds_ocean, ds_atmo)
    template, invariant, ocean_nn, atmo_nn = P._make_template(
        ds_ocean,
        ds_atmo,
        output_grid,
        P.DEFAULT_VERTICAL_COARSENING_INDICES,
        list(ds_ocean.time.values),
    )
    static = {
        "invariant": invariant.load(),
        "ocean_nn": ocean_nn,
        "atmo_nn": atmo_nn,
        "lat": ds_ocean["lat"].values,
        "lon": ds_ocean["lon"].values,
        "z_l": ds_ocean[P.VDIM].values,
        "variables": {
            k: (v.attrs, str(v.dtype))
            for k, v in template.data_vars.items()
            if "time" in v.dims
        },
        "output_grid": output_grid,
    }
    with open(os.path.join(work_dir, f"static-{output_grid}.pkl"), "wb") as f:
        pickle.dump(static, f)
    _regridder(work_dir, static)  # writes the weight files
    logging.info("setup done: %d time-varying variables", len(static["variables"]))


def _regridder(work_dir, static):
    g = tripolar.mom6_static_from_hgrid(
        xr.open_dataset(os.path.join(work_dir, "ocean_hgrid.nc"))
    )
    return tripolar.TripolarRegridder(g, static["lat"], static["lon"], work_dir)


def ocean_dataset(path, t, R, static) -> xr.Dataset:
    """One raw MOM6 file as a one-timestep dataset with mom6.zarr's layout."""
    raw = xr.open_dataset(path, decode_times=False, engine="h5netcdf").isel(time=0)
    z_l = raw["z_l"].values
    if not np.allclose(z_l, static["z_l"]):
        raise ValueError("raw file z_l differs from the zarr's")
    out = {}
    for v in SCALARS_3D + SCALARS_2D:
        out[v] = R.scalar(raw[v].load())
    out["ho"] = R.scalar(raw["ho"].load())
    for u, v in (("uo", "vo"), ("taux", "tauy")):
        out[u], out[v] = R.vector(raw[u].load(), raw[v].load())
    ds = xr.Dataset({k: v.astype(np.float32) for k, v in out.items()})
    ds = ds.assign_coords(z_l=z_l, zl=raw["zl"].values)
    ds = ds.expand_dims(time=[np.datetime64(t, "ns")])
    raw.close()
    return ds.transpose("time", ...)


def atmo_dataset(paths, times) -> xr.Dataset:
    parts = []
    for p, t in zip(paths, times):
        b = xr.open_dataset(p, decode_times=False, engine="h5netcdf")
        d = (
            b[ATMO_VARS]
            .isel(time=0)
            .load()
            .rename({"grid_xt": "lon", "grid_yt": "lat"})
        )
        valid = pd.Timestamp(
            b["time"].attrs["units"].split("since")[1].strip()
        ) + pd.Timedelta(hours=float(b["time"].values[0]))
        if valid != pd.Timestamp(t):
            raise ValueError(f"{p}: valid time {valid} != expected {t}")
        b.close()
        parts.append(
            d.reset_coords(drop=True).expand_dims(time=[np.datetime64(t, "ns")])
        )
    return xr.concat(parts, "time")


_STATE: dict = {}


def _init_worker(work_dir, output_grid, out_path, scratch):
    static = pickle.load(
        open(os.path.join(work_dir, f"static-{output_grid}.pkl"), "rb")
    )
    _STATE.update(
        static=static,
        R=_regridder(work_dir, static),
        fs=_s3(),
        out=out_path,
        scratch=scratch,
        grid=output_grid,
    )


def process_time(t: datetime.datetime) -> xr.Dataset:
    S = _STATE
    static, fs = S["static"], S["fs"]
    tmp = tempfile.mkdtemp(dir=S["scratch"])
    try:
        ocn = os.path.join(tmp, "ocn.nc")
        _fetch(fs, ocean_key(t), ocn)
        ds_o = P._clean_ocean_dataset(ocean_dataset(ocn, t, S["R"], static))
        os.remove(ocn)
        ocean = P._process_ocean_chunk(
            ds_o,
            S["grid"],
            P.DEFAULT_VERTICAL_COARSENING_INDICES,
            P._get_source_grid(ds_o),
            static["invariant"],
            static["ocean_nn"],
        )
        del ds_o
        times = [t - datetime.timedelta(hours=3), t]
        paths = []
        for i, ta in enumerate(times):
            paths.append(os.path.join(tmp, f"bfg{i}.nc"))
            _fetch(fs, atmo_key(ta), paths[-1])
        ds_a = atmo_dataset(paths, times)
        atmo = P._process_atmo_chunk(
            ds_a,
            S["grid"],
            P._get_source_grid(ds_a),
            static["invariant"],
            static["atmo_nn"],
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return xr.merge([ocean, atmo])


def _write_time(args):
    i, t = args
    t0 = time.time()
    try:
        ds = process_time(t)
        missing = set(_STATE["static"]["variables"]) - set(ds.data_vars)
        if missing:
            raise ValueError(f"missing output variables {sorted(missing)}")
        ds = ds[list(_STATE["static"]["variables"])].drop_vars(
            ["lat", "lon"], errors="ignore"
        )
        ds.to_zarr(_STATE["out"], region={"time": slice(i, i + 1)}, mode="r+")
        open(os.path.join(_STATE["out"], ".done", f"{i:06d}"), "w").close()
        return i, t, time.time() - t0, None
    except Exception as e:  # report, keep the pool alive
        return i, t, time.time() - t0, f"{type(e).__name__}: {e}"


def init_store(out_path, static, times):
    import dask.array as da

    inv = static["invariant"]
    nlat, nlon = inv.sizes["lat"], inv.sizes["lon"]
    ds = xr.Dataset(
        coords={"time": pd.DatetimeIndex(times), "lat": inv["lat"], "lon": inv["lon"]}
    )
    for name, (attrs, dtype) in static["variables"].items():
        ds[name] = (
            ("time", "lat", "lon"),
            da.zeros((len(times), nlat, nlon), chunks=(1, nlat, nlon), dtype=dtype),
            attrs,
        )
    for name in inv.data_vars:
        ds[name] = inv[name]
    enc = {
        n: {"chunks": (1, nlat, nlon), "_FillValue": np.nan}
        for n in static["variables"]
    }
    ds.to_zarr(out_path, mode="w", compute=False, encoding=enc, zarr_format=3)
    ds[list(inv.data_vars)].to_zarr(out_path, mode="a")
    os.makedirs(os.path.join(out_path, ".done"), exist_ok=True)


def run(work_dir, out_path, start, end, workers, output_grid, scratch):
    static = pickle.load(
        open(os.path.join(work_dir, f"static-{output_grid}.pkl"), "rb")
    )
    times = list(pd.date_range(start, end, freq="6h").to_pydatetime())
    if not os.path.exists(os.path.join(out_path, ".done")):
        init_store(out_path, static, times)
    done = {int(x) for x in os.listdir(os.path.join(out_path, ".done"))}
    todo = [(i, t) for i, t in enumerate(times) if i not in done]
    logging.info("%d of %d timesteps to do", len(todo), len(times))
    os.makedirs(scratch, exist_ok=True)
    failed = []
    with multiprocessing.get_context("spawn").Pool(
        workers,
        initializer=_init_worker,
        initargs=(work_dir, output_grid, out_path, scratch),
    ) as pool:
        for n, (i, t, dt, err) in enumerate(pool.imap_unordered(_write_time, todo)):
            if err:
                failed.append((str(t), err))
                logging.error("FAILED %s: %s", t, err)
            elif n % 20 == 0:
                logging.info("wrote %s (%.0f s), %d/%d", t, dt, n + 1, len(todo))
    json.dump(failed, open(os.path.join(out_path, ".failed.json"), "w"))
    if failed:
        raise SystemExit(f"{len(failed)} timesteps failed; rerun to retry them")
    logging.info("all %d timesteps written", len(times))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("work_dir")
    r = sub.add_parser("run")
    r.add_argument("work_dir")
    r.add_argument("out")
    r.add_argument("start")
    r.add_argument("end")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    for q in (s, r):
        q.add_argument("--output_grid", default=P.DEFAULT_OUTPUT_GRID)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup(a.work_dir, a.output_grid)
    else:
        run(a.work_dir, a.out, a.start, a.end, a.workers, a.output_grid, a.scratch)
