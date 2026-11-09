# ruff: noqa: E501
"""Build and upload the 1deg static ocean forcing for coupled abrupt-4xCO2 inference.

Same recipe as the 4deg files in
gs://vcm-ml-intermediate/2026-10-20-cm4-abrupt-4xCO2-ensemble-4deg-static-ocean-forcing:
one netCDF per ensemble member holding the ocean's time-constant forcings
(land_fraction, sea_surface_fraction, deptho, hfgeou), copied from the 1deg
piControl daily stores and repeated on a 30-record 5-day axis that starts at
the member's initial time (see make_configs.py for the start times).

Before writing, checks that
- the static fields match the 5-daily store the coupled fine-tunes trained on;
- the IC and atmosphere forcing stores cover every member's 90-day window and
  the 100-year extension (after the configs' n_repeats).

Usage: python make_static_ocean_forcing.py <local_output_dir> [--upload]
"""

import argparse
import base64
import datetime
import hashlib
import os

import cftime
import gcsfs
import numpy as np
import xarray as xr
from make_configs import start_times

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
N_TIMES = 30


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


def _check_against_training(arrays):
    stores = {k: xr.open_zarr(v) for k, v in TRAINING_5DAILY.items()}
    for n, a in arrays.items():
        src = next(s for s in stores.values() if n in s)
        v = src[n]
        b = (v.isel(time=0) if "time" in v.dims else v).values
        same = np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(
            a, b, equal_nan=True
        )
        print(f"  {n}: matches training 5-daily store: {same}")
        assert same, n


def _check_coverage(times):
    starts = times
    last_member_end = max(starts) + datetime.timedelta(days=90)
    extension_end = cftime.datetime(401, 1, 1, calendar="noleap")
    ic_stores = {"ocean IC": OCEAN_DAILY, "atmosphere IC": ATMOS_IC}
    for name, path in ic_stores.items():
        idx = xr.open_zarr(path).time.to_index()
        missing = [t for t in starts if t not in idx]
        print(f"  {name}: {idx[0]} to {idx[-1]}; start times missing: {missing}")
        assert not missing, name
    for path in ATMOS_FORCING:
        idx = xr.open_zarr(path).time.to_index()
        step = idx[1] - idx[0]
        # end of the 100-year extension's forcing with n_repeats: 2
        repeated_end = idx[0] + 2 * (idx[-1] - idx[0] + step) - step
        print(
            f"  forcing {path.rsplit('/', 1)[-1]}: {idx[0]} to {idx[-1]} "
            f"(step {step}); repeated twice ends {repeated_end}"
        )
        assert idx[0] <= min(starts) and idx[-1] >= last_member_end, path
        assert repeated_end >= extension_end, path


def build(out):
    os.makedirs(out, exist_ok=True)
    times = start_times()
    arrays, attrs, co = _static_arrays()
    print("Checking static fields against the training store:")
    _check_against_training(arrays)
    print("Checking store time coverage:")
    _check_coverage(times)
    for member, t0 in enumerate(times, start=1):
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
            f"stores) on a 5-day axis from {t[0]} for coupled abrupt-4xCO2 "
            f"ensemble member {member}."
        )
        ds.to_netcdf(os.path.join(out, f"ic_{member:04d}.nc"))
    for member in range(1, len(times) + 1):
        x = xr.open_dataset(os.path.join(out, f"ic_{member:04d}.nc"))
        for n, a in arrays.items():
            b = x[n].values
            assert b.shape[0] == N_TIMES, (member, n)
            assert np.array_equal(np.isnan(b[0]), np.isnan(a)), (member, n)
            assert np.allclose(b, a[None], equal_nan=True), (member, n)
    nans = {n: int(np.isnan(a).sum()) for n, a in arrays.items()}
    print(
        f"wrote and verified {len(times)} files; shape {arrays['deptho'].shape}; NaN counts {nans}"
    )


def upload(out):
    fs = gcsfs.GCSFileSystem()
    if fs.exists(DST):
        raise SystemExit(f"gs://{DST} already exists; not overwriting")
    files = sorted(f for f in os.listdir(out) if f.endswith(".nc"))
    for f in files:
        fs.put_file(os.path.join(out, f), f"{DST}/{f}")
    fs.invalidate_cache()

    def md5(p):
        return base64.b64encode(hashlib.md5(open(p, "rb").read()).digest()).decode()

    ok = all(
        fs.info(f"{DST}/{f}").get("md5Hash") == md5(os.path.join(out, f)) for f in files
    )
    print(f"uploaded {len(files)} files to gs://{DST}; md5 match: {ok}")
    assert ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    build(args.out)
    if args.upload:
        upload(args.out)
