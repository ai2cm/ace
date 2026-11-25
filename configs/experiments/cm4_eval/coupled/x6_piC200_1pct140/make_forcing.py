# ruff: noqa: E501
"""Build and upload the forcing for the X6 340-year piControl-then-1pctCO2 run.

inference-config-piC200-1pct140.yaml runs 17 segments of 20 years from the
piControl state at 0151-01-06: 200 years at fixed CO2, then 140 years of 1%/yr
CO2 growth, with model time continuous to 0491-01-06. Writes two zarr stores:

- atmosphere.zarr: carbon_dioxide(time) on a 6-hourly noleap axis with the CM4
  atmosphere stores' phase (first time 0151-01-01T06): the piControl store's
  values through 0351-01-01T00, the 1pctCO2 store's values shifted +350 years
  (0351-01-01T06 to 0491-01-01T00), then 1%/yr compounded growth to END. Plus
  the time-invariant land_fraction, lake_fraction and HGTsfc as (lat, lon)
  arrays, each from the store X6's training merge took it from (see
  ATMOS_TRAINING). DSWRFtoa is an all-NaN (lat, lon) placeholder: the X6
  atmosphere derives it (stepper.derived_forcings.insolation) and overwrites
  the loaded field before use, but the coupled forcing loader still requires
  the variable (CoupledStepperConfig.get_forcing_window_data_requirements does
  not drop derived forcings). NaN makes any read of the placeholder visible.
- ocean.zarr: a 5-day time coordinate from 0151-01-06 to END and the
  time-invariant land_fraction, sea_surface_fraction, deptho and hfgeou as
  (lat, lon) arrays. Statics-only stores need ace #1663.

Both are zarr because the forcing is read from gs:// (job_runner mounts only
weka /climate-default), and XarrayDataset opens remote files only with the
zarr engine.

Before writing, checks that
- the statics equal those of the stores X6 trained on (NaN pattern included);
- the initial-condition stores contain 0151-01-06T00;
- the CO2 time axis is strictly 6-hourly and the 0351 splice jumps < 0.01 ppm;
- with --checkpoint, the checkpoint's ocean vertical coordinate carries deptho
  equal to the store's.

Usage: python make_forcing.py <local_output_dir> [--checkpoint <ckpt.tar>] [--upload]
"""

import argparse
import base64
import datetime
import hashlib
import os
import tempfile

import cftime
import gcsfs
import numpy as np
import xarray as xr

BUCKET = "vcm-ml-intermediate"
DST = f"{BUCKET}/2026-11-24-x6-piC200-1pct140-forcing"
ATMOS_PIC = (
    f"gs://{BUCKET}/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr.zarr"
)
ATMOS_1PCT = (
    f"gs://{BUCKET}/2026-06-19-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr.zarr"
)
# X6 trained on merges of [coupled-atmosphere store, CM4 atmosphere store] for
# each of piControl and 1pctCO2. A merge takes each variable from the first
# store that has it, so land_fraction came from the coupled-atmosphere stores
# (which differ from the CM4 atmosphere stores' land_fraction), and
# lake_fraction and HGTsfc from the CM4 atmosphere stores.
ATMOS_TRAINING = {
    "piControl": [
        f"gs://{BUCKET}/2026-07-15-om4-picontrol-1deg-coupled-atmosphere.zarr",
        ATMOS_PIC,
    ],
    "1pctCO2": [
        f"gs://{BUCKET}/2026-07-15-om4-1pctco2-1deg-coupled-atmosphere.zarr",
        ATMOS_1PCT,
    ],
}
COUPLED_DAILY = f"gs://{BUCKET}/2026-09-02-cm4-piControl-1deg-coupled-1daily-ocean.zarr"
OCEAN_DAILY = f"gs://{BUCKET}/2026-09-02-cm4-piControl-1deg-ocean-1daily.zarr"
OCEAN_TRAINING_5DAILY = [
    f"gs://{BUCKET}/2026-07-15-om4-picontrol-1deg-coupled-ocean.zarr",
    f"gs://{BUCKET}/2026-07-15-om4-picontrol-1deg-ocean-5daily.zarr",
]
ATMOS_STATICS = ["land_fraction", "lake_fraction", "HGTsfc"]
INSOLATION_NAME = "DSWRFtoa"
OCEAN_STATICS = ["land_fraction", "sea_surface_fraction", "deptho", "hfgeou"]

IC_TIME = cftime.DatetimeNoLeap(151, 1, 6)
SPLICE_TIME = cftime.DatetimeNoLeap(351, 1, 1)
SHIFT_YEARS = 350
# 340 years of 5-day steps from IC_TIME end at 0491-01-06; END leaves margin.
RUN_END = cftime.DatetimeNoLeap(491, 1, 6)
END = cftime.DatetimeNoLeap(491, 1, 31)
SIX_HOURS = datetime.timedelta(hours=6)
FIVE_DAYS = datetime.timedelta(days=5)
SPLICE_TOLERANCE_PPM = 0.01


def _same(a, b):
    return (
        a.shape == b.shape
        and np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
    )


def _static(ds, name, n_samples=5):
    """Return ds[name] as a 2D array; if it has a time dim, assert it is constant."""
    v = ds[name]
    if "time" not in v.dims:
        return v.values
    idx = np.linspace(0, v.sizes["time"] - 1, n_samples).astype(int)
    samples = v.isel(time=idx).values
    for s in samples[1:]:
        assert _same(samples[0], s), f"{name} varies in time"
    return samples[0]


def _shift_years(t, years):
    return cftime.DatetimeNoLeap(
        t.year + years, t.month, t.day, t.hour, t.minute, t.second
    )


def _check_axis(times, step, name):
    d = np.array([b - a for a, b in zip(times[:-1], times[1:])])
    assert np.all(d == step), f"{name}: time axis is not strictly {step}"


def _open_local_copy(store, names, tmp):
    """Open a local copy of `names` from a zarr store.

    The CM4 stores shard carbon_dioxide with one-element inner chunks, so
    reading it from GCS takes one range request per time; whole-shard
    downloads are much faster.
    """
    fs = gcsfs.GCSFileSystem()
    src = store.removeprefix("gs://")
    dst = os.path.join(tmp, os.path.basename(src))
    os.makedirs(dst)
    fs.get(f"{src}/zarr.json", f"{dst}/zarr.json")
    for n in names:
        fs.get(f"{src}/{n}", f"{dst}/{n}", recursive=True)
    return xr.open_zarr(dst)


def _co2():
    tmp = tempfile.mkdtemp()
    names = ["carbon_dioxide", "time"]
    pic = _open_local_copy(ATMOS_PIC, names, tmp)
    one = _open_local_copy(ATMOS_1PCT, names, tmp)
    pic_t = list(pic.time.values)
    one_t = [_shift_years(t, SHIFT_YEARS) for t in one.time.values]
    pic_v = pic["carbon_dioxide"].values.astype(np.float64)
    one_v = one["carbon_dioxide"].values.astype(np.float64)
    assert pic_t[-1] == SPLICE_TIME, pic_t[-1]
    assert one_t[0] == SPLICE_TIME + SIX_HOURS, one_t[0]
    n_ext = int((END - one_t[-1]) / SIX_HOURS)
    ext_t = [one_t[-1] + (i + 1) * SIX_HOURS for i in range(n_ext)]
    # 1%/yr compounded at each 6-hour step (1460 steps per noleap year).
    ext_v = one_v[-1] * 1.01 ** (np.arange(1, n_ext + 1) / 1460.0)
    times = pic_t + one_t + ext_t
    values = np.concatenate([pic_v, one_v, ext_v])
    _check_axis(times, SIX_HOURS, "carbon_dioxide")
    assert len(set(times)) == len(times)
    assert np.all(np.isfinite(values))
    jump = one_v[0] - pic_v[-1]
    print(
        f"  piC CO2 {pic_v.min():.3f}..{pic_v.max():.3f} ppm; 1pct CO2 {one_v[0]:.3f}"
        f"..{one_v[-1]:.3f}; splice jump at {SPLICE_TIME}: {jump:+.4f} ppm"
    )
    assert abs(jump) < SPLICE_TOLERANCE_PPM, jump
    # The 1pct store's own growth rate, to compare with the extension's.
    per_year = (one_v[-1] / one_v[-1 - 1460]) - 1
    print(f"  1pct store's last-year growth: {100 * per_year:.4f}%/yr")
    attrs = dict(pic["carbon_dioxide"].attrs)
    attrs.pop("_FillValue", None)
    return times, values.astype(np.float32), attrs


def _atmos_statics():
    """Return the atmosphere statics as X6 saw them in training (merge order)."""
    arrays, attrs = {}, {}
    for experiment, paths in ATMOS_TRAINING.items():
        stores = [(p, xr.open_zarr(p)) for p in paths]
        for n in ATMOS_STATICS:
            path, ds = next((p, d) for p, d in stores if n in d)
            a = _static(ds, n)
            name = path.rsplit("/", 1)[-1]
            if n not in arrays:
                arrays[n], attrs[n] = a, ds[n].attrs
                print(f"  atmosphere {n}: from {name}")
            else:
                same = _same(arrays[n], a)
                print(
                    f"  atmosphere {n}: matches {experiment} training ({name}): {same}"
                )
                assert same, (n, experiment)
    return arrays, attrs


def _ocean_statics():
    co = xr.open_zarr(COUPLED_DAILY)
    oc = xr.open_zarr(OCEAN_DAILY)
    # Raw arrays: stores' lat/lon can differ at the 1e-14 level, and xarray
    # alignment would turn the mismatch into NaNs.
    arrays = {n: _static(co, n) for n in ["land_fraction", "sea_surface_fraction"]}
    arrays["deptho"] = _static(co, "deptho")
    arrays["hfgeou"] = _static(oc, "hfgeou")
    attrs = {n: (oc if n == "hfgeou" else co)[n].attrs for n in arrays}
    training = [xr.open_zarr(p) for p in OCEAN_TRAINING_5DAILY]
    for n, a in arrays.items():
        src = next(s for s in training if n in s)
        same = _same(a, _static(src, n))
        print(f"  ocean {n}: matches training 5-daily store: {same}")
        assert same, n
    return arrays, attrs, co


def _check_ic():
    for path in [OCEAN_DAILY, ATMOS_PIC]:
        idx = xr.open_zarr(path).time.to_index()
        print(f"  {path.rsplit('/', 1)[-1]}: contains {IC_TIME}: {IC_TIME in idx}")
        assert IC_TIME in idx, path


def _check_coords(atmos, ocean):
    for c in ["lat", "lon"]:
        a, o = atmos[c].values, ocean[c].values
        print(f"  {c}: atmosphere vs ocean max |diff| {np.abs(a - o).max():.2e}")
        assert np.allclose(a, o, atol=1e-10), c


def check_checkpoint(path, deptho):
    """Assert the checkpoint's ocean vertical coordinate carries deptho == store's.

    Checks both copies of the ocean dataset_info in the coupled checkpoint:
    stepper.dataset_info.ocean and stepper.ocean_state.dataset_info.
    """
    import torch

    stepper = torch.load(path, map_location="cpu", weights_only=False)["stepper"]
    infos = {
        "dataset_info.ocean": stepper["dataset_info"]["ocean"],
        "ocean_state.dataset_info": stepper["ocean_state"]["dataset_info"],
    }
    insolation = stepper["config"]["atmosphere"]["stepper"]["derived_forcings"][
        "insolation"
    ]
    print(f"  checkpoint derives {insolation['insolation_name']} (insolation)")
    assert insolation["insolation_name"] == INSOLATION_NAME, insolation
    st = np.asarray(deptho, dtype=np.float64)
    for where, info in infos.items():
        vc = info["vertical_coordinate"]
        assert "deptho" in vc, f"{where}.vertical_coordinate has keys {list(vc)}"
        ck = vc["deptho"].cpu().numpy().astype(np.float64)
        assert ck.shape == st.shape, (where, ck.shape, st.shape)
        same = _same(ck, st)
        print(
            f"  checkpoint {where} deptho {ck.shape}: equal incl. NaN pattern: {same}"
        )
        assert same, where


def build(out, checkpoint=None):
    os.makedirs(out, exist_ok=True)
    print("Checking initial-condition stores:")
    _check_ic()
    print("Building carbon_dioxide:")
    co2_t, co2_v, co2_attrs = _co2()
    pic = xr.open_zarr(ATMOS_PIC)
    print("Checking atmosphere statics:")
    a_arrays, a_attrs = _atmos_statics()
    print("Checking ocean statics:")
    o_arrays, o_attrs, co = _ocean_statics()
    print("Checking lat/lon:")
    _check_coords(pic, co)
    if checkpoint is not None:
        print("Checking the checkpoint's deptho and derived insolation:")
        check_checkpoint(checkpoint, o_arrays["deptho"])
    else:
        print(
            "WARNING: no --checkpoint; checkpoint deptho and derived insolation "
            "not checked"
        )

    atmos = xr.Dataset(
        {
            "carbon_dioxide": (("time",), co2_v, co2_attrs),
            **{
                n: (("lat", "lon"), a.astype("float32"), a_attrs[n])
                for n, a in a_arrays.items()
            },
            INSOLATION_NAME: (
                ("lat", "lon"),
                np.full(a_arrays["land_fraction"].shape, np.nan, dtype="float32"),
                {
                    "long_name": "placeholder, never read: the X6 atmosphere "
                    "derives DSWRFtoa (derived_forcings.insolation)"
                },
            ),
        },
        coords={"time": co2_t, "lat": pic.lat.values, "lon": pic.lon.values},
    )
    atmos.attrs["description"] = (
        "CO2 for the X6 340-year run (CM4 piControl to 0351-01-01T00, then CM4 "
        "1pctCO2 shifted +350 years, then 1%/yr growth) and time-invariant "
        "atmosphere statics from the CM4 piControl store. Built by "
        "configs/experiments/cm4_eval/coupled/x6_piC200_1pct140/make_forcing.py."
    )
    n_ocean = int((END - IC_TIME) / FIVE_DAYS) + 1
    ocean_t = [IC_TIME + i * FIVE_DAYS for i in range(n_ocean)]
    assert ocean_t[-1] == END and RUN_END in ocean_t
    ocean = xr.Dataset(
        {
            n: (("lat", "lon"), a.astype("float32"), o_attrs[n])
            for n, a in o_arrays.items()
        },
        coords={"time": ocean_t, "lat": co.lat.values, "lon": co.lon.values},
    )
    ocean.attrs["description"] = (
        "Time-invariant ocean forcing (CM4 piControl 1deg daily stores) on a 5-day "
        f"axis from {IC_TIME} to {END} for the X6 340-year run. Built by "
        "configs/experiments/cm4_eval/coupled/x6_piC200_1pct140/make_forcing.py."
    )
    time_encoding = {"units": "hours since 0151-01-01 00:00:00", "dtype": "int64"}
    for ds in (atmos, ocean):
        for v in ds.variables.values():
            v.encoding = {}
        ds["time"].encoding.update(time_encoding, calendar="noleap")
    atmos["carbon_dioxide"].encoding["chunks"] = (1460 * 20,)
    atmos.to_zarr(os.path.join(out, "atmosphere.zarr"), mode="w-")
    ocean.to_zarr(os.path.join(out, "ocean.zarr"), mode="w-")
    verify(out, co2_t, co2_v, a_arrays, ocean_t, o_arrays)


def verify(out, co2_t, co2_v, a_arrays, ocean_t, o_arrays):
    a = xr.open_zarr(os.path.join(out, "atmosphere.zarr"), use_cftime=True)
    o = xr.open_zarr(os.path.join(out, "ocean.zarr"), use_cftime=True)
    assert list(a.time.values) == list(co2_t)
    assert np.array_equal(a.carbon_dioxide.values, co2_v)
    assert list(o.time.values) == list(ocean_t)
    _check_axis(list(o.time.values), FIVE_DAYS, "ocean time")
    for n, x in a_arrays.items():
        assert _same(a[n].values, x.astype("float32")), n
    assert np.isnan(a[INSOLATION_NAME].values).all()
    for n, x in o_arrays.items():
        assert _same(o[n].values, x.astype("float32")), n
    co2 = a.carbon_dioxide
    print(f"wrote and verified {out}")
    print(
        f"  atmosphere: {a.time.values[0]} to {a.time.values[-1]} ({a.sizes['time']} times)"
    )
    print(
        f"  ocean: {o.time.values[0]} to {o.time.values[-1]} ({o.sizes['time']} times)"
    )
    for t in [
        "0351-01-01T00:00:00",
        "0351-01-01T06:00:00",
        "0351-01-06T00:00:00",
        "0421-01-01T00:00:00",
        "0491-01-01T00:00:00",
        "0491-01-06T00:00:00",
        "0491-01-11T00:00:00",
    ]:
        print(f"  carbon_dioxide at {t}: {float(co2.sel(time=t)):.4f} ppm")


def _md5(p):
    return base64.b64encode(hashlib.md5(open(p, "rb").read()).digest()).decode()


def upload(out):
    fs = gcsfs.GCSFileSystem()
    if fs.exists(DST):
        raise SystemExit(f"gs://{DST} already exists; not overwriting")
    files = []
    for root, _, names in os.walk(out):
        for f in names:
            local = os.path.join(root, f)
            files.append((local, f"{DST}/{os.path.relpath(local, out)}"))
    for local, remote in files:
        fs.put_file(local, remote)
    fs.invalidate_cache()
    bad = [r for loc, r in files if fs.info(r).get("md5Hash") != _md5(loc)]
    total = sum(os.path.getsize(loc) for loc, _ in files)
    print(
        f"uploaded {len(files)} files ({total / 1e6:.1f} MB) to gs://{DST}; "
        f"md5 mismatches: {bad}"
    )
    assert not bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--checkpoint", help="local copy of the X6 checkpoint")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    build(args.out, args.checkpoint)
    if args.upload:
        upload(args.out)
