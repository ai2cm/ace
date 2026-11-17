# ruff: noqa: E501
"""Build and upload the 1deg static ocean forcing for coupled abrupt-4xCO2 inference.

Same recipe as the 4deg files in
gs://vcm-ml-intermediate/2026-10-20-cm4-abrupt-4xCO2-ensemble-4deg-static-ocean-forcing:
one netCDF per initial time holding the ocean's time-constant forcings
(land_fraction, sea_surface_fraction, deptho, hfgeou), copied from the 1deg
piControl daily stores and repeated on a 30-record 5-day axis that starts at
the initial time (see make_configs.py for the start times).

Two sets of files are built, each uploaded to its own directory:
- ensemble: one file per piControl-branched ensemble member (ic_0001.nc, ...);
- abrupt-run: one file for the 100-year run initialized from CM4's own
  abrupt-4xCO2 simulation at 0251-01-02 (ic_0251-01-02.nc).

Before writing, checks that
- the static fields match the 5-daily store the coupled fine-tunes trained on
  (and, for abrupt-run, the abrupt-4xCO2 initial-condition ocean stores);
- the IC stores contain every start time, and the atmosphere forcing stores
  cover every run (after the configs' n_repeats).

Usage: python make_static_ocean_forcing.py <local_output_dir>
    [--set {ensemble,abrupt-run}] [--upload]
"""

import argparse
import base64
import dataclasses
import datetime
import hashlib
import os

import cftime
import gcsfs
import numpy as np
import xarray as xr
from make_configs import (
    ABRUPT_RUN_START,
    ABRUPT_RUN_STATIC_FORCING,
    abrupt_run_forcing_file,
    start_times,
)

BUCKET = "vcm-ml-intermediate"
DST = f"{BUCKET}/2026-11-05-cm4-abrupt-4xCO2-ensemble-1deg-static-ocean-forcing"
COUPLED_DAILY = f"gs://{BUCKET}/2026-09-02-cm4-piControl-1deg-coupled-1daily-ocean.zarr"
OCEAN_DAILY = f"gs://{BUCKET}/2026-09-02-cm4-piControl-1deg-ocean-1daily.zarr"
TRAINING_5DAILY = {
    "coupled": f"gs://{BUCKET}/2026-07-15-om4-picontrol-1deg-coupled-ocean.zarr",
    "ocean": f"gs://{BUCKET}/2026-07-15-om4-picontrol-1deg-ocean-5daily.zarr",
}
ATMOS_IC = (
    f"gs://{BUCKET}/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr"
)
ATMOS_FORCING = [
    f"gs://{BUCKET}/2026-07-15-om4-picontrol-1deg-coupled-atmosphere.zarr",
    ATMOS_IC,
]
ABRUPT_RUN_DST = f"{BUCKET}/{os.path.basename(ABRUPT_RUN_STATIC_FORCING)}"
ABRUPT_RUN_IC = {
    "ocean IC": f"gs://{BUCKET}/2026-11-17-cm4-cmip6-abrupt-4xCO2-initial-condition-1deg-ocean-1daily.zarr",
    "atmosphere IC": f"gs://{BUCKET}/2026-11-17-CM4-CMIP6-abrupt-4xCO2-initial-condition-atmosphere-land-1deg-8layer-2day.zarr",
}
ABRUPT_RUN_COUPLED_OCEAN = f"gs://{BUCKET}/2026-11-17-cm4-cmip6-abrupt-4xCO2-initial-condition-1deg-coupled-1daily-ocean.zarr"
N_TIMES = 30
# 7300 5-day coupled steps, as in make_configs.HUNDRED_YEAR
HUNDRED_YEARS = datetime.timedelta(days=5 * 7300)


@dataclasses.dataclass
class ForcingSet:
    """Files to build and the stores they must be consistent with.

    files: (file name, initial time, description of the run it is for).
    ic_stores: IC stores that must contain every initial time.
    end_without_repeats: latest time the forcing stores must cover unrepeated.
    end_with_repeats: latest time the forcing stores must cover with n_repeats: 2.
    extra_static_stores: further stores whose static fields must match, if
        they hold them.
    """

    dst: str
    files: list[tuple[str, cftime.datetime, str]]
    ic_stores: dict[str, str]
    end_without_repeats: cftime.datetime | None
    end_with_repeats: cftime.datetime
    extra_static_stores: list[str]


def _ensemble_set() -> ForcingSet:
    times = start_times()
    return ForcingSet(
        dst=DST,
        files=[
            (f"ic_{m:04d}.nc", t, f"coupled abrupt-4xCO2 ensemble member {m}")
            for m, t in enumerate(times, start=1)
        ],
        ic_stores={"ocean IC": OCEAN_DAILY, "atmosphere IC": ATMOS_IC},
        end_without_repeats=max(times) + datetime.timedelta(days=90),
        end_with_repeats=cftime.datetime(401, 1, 1, calendar="noleap"),
        extra_static_stores=[],
    )


def _abrupt_run_set() -> ForcingSet:
    return ForcingSet(
        dst=ABRUPT_RUN_DST,
        files=[
            (
                abrupt_run_forcing_file(),
                ABRUPT_RUN_START,
                "the 100-year coupled run initialized from CM4's abrupt-4xCO2 "
                "simulation",
            )
        ],
        ic_stores=ABRUPT_RUN_IC,
        end_without_repeats=None,
        end_with_repeats=ABRUPT_RUN_START + HUNDRED_YEARS,
        extra_static_stores=[ABRUPT_RUN_COUPLED_OCEAN, ABRUPT_RUN_IC["ocean IC"]],
    )


FORCING_SETS = {"ensemble": _ensemble_set, "abrupt-run": _abrupt_run_set}


def _static_arrays():
    co = xr.open_zarr(COUPLED_DAILY)
    oc = xr.open_zarr(OCEAN_DAILY)

    def first(v):
        return (v.isel(time=0) if "time" in v.dims else v).values

    # Build from raw arrays: stores' lat/lon can differ at the 1e-14 level, and
    # xarray alignment would turn the mismatch into NaNs.
    arrays = {
        n: first(co[n]) for n in ["land_fraction", "sea_surface_fraction", "deptho"]
    }
    arrays["hfgeou"] = first(oc["hfgeou"])
    attrs = {n: (oc if n == "hfgeou" else co)[n].attrs for n in arrays}
    for n, a in arrays.items():
        assert a.shape == (co.sizes["lat"], co.sizes["lon"]), (n, a.shape)
    return arrays, attrs, co


def _first_time(v: xr.DataArray) -> np.ndarray:
    return (v.isel(time=0) if "time" in v.dims else v).values


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(
        np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, equal_nan=True)
    )


def _check_against_training(arrays):
    stores = {k: xr.open_zarr(v) for k, v in TRAINING_5DAILY.items()}
    for n, a in arrays.items():
        src = next(s for s in stores.values() if n in s)
        same = _same(a, _first_time(src[n]))
        print(f"  {n}: matches training 5-daily store: {same}")
        assert same, n


def _check_against_stores(arrays, paths):
    for path in paths:
        ds = xr.open_zarr(path)
        for n, a in arrays.items():
            if n not in ds:
                print(f"  {n}: not in {path.rsplit('/', 1)[-1]}")
                continue
            same = _same(a, _first_time(ds[n]))
            print(f"  {n}: matches {path.rsplit('/', 1)[-1]}: {same}")
            assert same, (n, path)


def _check_coverage(forcing_set: ForcingSet):
    starts = [t for _, t, _ in forcing_set.files]
    for name, path in forcing_set.ic_stores.items():
        idx = xr.open_zarr(path).time.to_index()
        missing = [t for t in starts if t not in idx]
        print(f"  {name}: {idx[0]} to {idx[-1]}; start times missing: {missing}")
        assert not missing, name
    for path in ATMOS_FORCING:
        idx = xr.open_zarr(path).time.to_index()
        step = idx[1] - idx[0]
        # end of the forcing with the configs' n_repeats: 2
        repeated_end = idx[0] + 2 * (idx[-1] - idx[0] + step) - step
        print(
            f"  forcing {path.rsplit('/', 1)[-1]}: {idx[0]} to {idx[-1]} "
            f"(step {step}); repeated twice ends {repeated_end}"
        )
        assert idx[0] <= min(starts), path
        if forcing_set.end_without_repeats is not None:
            assert idx[-1] >= forcing_set.end_without_repeats, path
        assert repeated_end >= forcing_set.end_with_repeats, path


def build(out, forcing_set: ForcingSet):
    os.makedirs(out, exist_ok=True)
    arrays, attrs, co = _static_arrays()
    print("Checking static fields against the training store:")
    _check_against_training(arrays)
    if forcing_set.extra_static_stores:
        print("Checking static fields against other stores:")
        _check_against_stores(arrays, forcing_set.extra_static_stores)
    print("Checking store time coverage:")
    _check_coverage(forcing_set)
    for name, t0, run in forcing_set.files:
        t = xr.date_range(
            t0, periods=N_TIMES, freq="5D", calendar="noleap", use_cftime=True
        )
        ds = xr.Dataset(
            {
                n: (
                    ("time", "lat", "lon"),
                    np.broadcast_to(a, (N_TIMES,) + a.shape).astype("float32"),
                    attrs[n],
                )
                for n, a in arrays.items()
            },
            coords={"time": t, "lat": co.lat.values, "lon": co.lon.values},
        )
        ds.attrs["description"] = (
            "Time-constant ocean forcing (copied from the CM4 piControl 1deg daily "
            f"stores) on a 5-day axis from {t[0]} for {run}."
        )
        ds.to_netcdf(os.path.join(out, name))
    for name, t0, _ in forcing_set.files:
        x = xr.open_dataset(os.path.join(out, name))
        assert x.time.values[0] == t0, (name, x.time.values[0])
        for n, a in arrays.items():
            b = x[n].values
            assert b.shape[0] == N_TIMES, (name, n)
            assert np.array_equal(np.isnan(b[0]), np.isnan(a)), (name, n)
            assert np.allclose(b, a[None], equal_nan=True), (name, n)
    nans = {n: int(np.isnan(a).sum()) for n, a in arrays.items()}
    print(
        f"wrote and verified {len(forcing_set.files)} files; shape {arrays['deptho'].shape}; NaN counts {nans}"
    )


def upload(out, forcing_set: ForcingSet):
    dst = forcing_set.dst
    fs = gcsfs.GCSFileSystem()
    if fs.exists(dst):
        raise SystemExit(f"gs://{dst} already exists; not overwriting")
    files = sorted(name for name, _, _ in forcing_set.files)
    for f in files:
        fs.put_file(os.path.join(out, f), f"{dst}/{f}")
    fs.invalidate_cache()

    def md5(p):
        return base64.b64encode(hashlib.md5(open(p, "rb").read()).digest()).decode()

    ok = all(
        fs.info(f"{dst}/{f}").get("md5Hash") == md5(os.path.join(out, f)) for f in files
    )
    print(f"uploaded {len(files)} files to gs://{dst}; md5 match: {ok}")
    assert ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--set", choices=sorted(FORCING_SETS), default="ensemble")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    forcing_set = FORCING_SETS[args.set]()
    build(args.out, forcing_set)
    if args.upload:
        upload(args.out, forcing_set)
