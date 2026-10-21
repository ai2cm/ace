"""Build the daily 4deg store of Spencer's re-derived SHiELD-SOM Q-flux.

Spencer re-derived the C96 SHiELD-SOM Q-flux climatology directly at 4deg
(QFLUX_CLIMATOLOGY: twelve monthly means stamped at each month's midpoint in
2031). The Q-flux in the SOM ensemble stores was regridded from C96 instead,
and is off in coastal, partly-land cells.

This writes that climatology as ``prescribed_qflux`` on the exact time axis of
the SOM ensemble stores (3653 daily labels, 2031-01-02T00 .. 2041-01-01T00), so
a ``merge:`` loader with this store listed first overrides the store's own
Q-flux (the first source that has a variable supplies it). Every store labels
a day's mean at the following 00Z, so each label gets the climatology linearly
interpolated in time to the middle of the day it averages (label - 12h), between
the month midpoints of the actual year (periodic across December-January).

The grid is copied from the template store; the climatology's must already
match it exactly. Writes OUTPUT_STORE on GCS; copy it to WEKA_STORE with
``scripts/data_process/gcs_to_weka.sh OUTPUT_STORE WEKA_STORE``.

``--abrupt-ensemble`` instead writes one store per member of SHiELD's
abrupt-4xCO2 ensemble, each on that member's own 90-day time axis (member
ic_00NN starts at its own initial month), to ABRUPT_ENSEMBLE_OUTPUT_DIR; a merge
needs every source to share its sample times. Copy the directory to
ABRUPT_ENSEMBLE_WEKA_DIR the same way.

Usage:
    python build_qflux_4deg_store.py [--abrupt-ensemble] [--dry-run]
"""

import argparse
import subprocess

import google.oauth2.credentials
import numpy as np
import pandas as pd
import xarray as xr

QFLUX_CLIMATOLOGY = (
    "gs://vcm-ml-intermediate/2026-10-20-C96-SHiELD-Q-flux-re-derived-at-4deg.zarr"
)
TEMPLATE_STORE = (
    "gs://vcm-ml-intermediate/"
    "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset/"
    "1xCO2-ic_0005.zarr"
)
# vcm-ml-intermediate is not writable with user credentials; the weka copy
# (WEKA_STORE) is what the configs read.
OUTPUT_STORE = (
    "gs://vcm-ml-scratch/alexeyy/"
    "2026-10-21-4deg-daily-c96-shield-som-qflux-rederived-at-4deg-fme-dataset/"
    "qflux.zarr"
)
WEKA_STORE = (
    "/climate-default/"
    "2026-10-21-4deg-daily-c96-shield-som-qflux-rederived-at-4deg-fme-dataset/"
    "qflux.zarr"
)
ABRUPT_ENSEMBLE_STORE_DIR = (
    "gs://vcm-ml-intermediate/"
    "2026-09-21-vertically-resolved-4deg-daily-c96-shield-som-abrupt-4xCO2-"
    "ensemble-fme-dataset"
)
ABRUPT_ENSEMBLE_OUTPUT_DIR = (
    "gs://vcm-ml-scratch/alexeyy/"
    "2026-10-21-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-qflux-rederived-"
    "at-4deg-fme-dataset"
)
ABRUPT_ENSEMBLE_WEKA_DIR = (
    "/climate-default/"
    "2026-10-21-4deg-daily-c96-shield-som-abrupt-4xCO2-ensemble-qflux-rederived-"
    "at-4deg-fme-dataset"
)
ABRUPT_ENSEMBLE_N_MEMBERS = 36
NAME = "prescribed_qflux"
HALF_DAY = pd.Timedelta(hours=12)


def _storage_options() -> dict:
    # Application-default credentials are not always fresh on a laptop; the
    # gcloud user token is.
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
    return {"token": google.oauth2.credentials.Credentials(token)}


def _month_midpoints(years: range) -> pd.DatetimeIndex:
    starts = pd.date_range(f"{years.start}-01-01", f"{years.stop}-01-01", freq="MS")
    return pd.DatetimeIndex(starts[:-1] + (starts[1:] - starts[:-1]) / 2)


def interpolate_climatology(
    climatology: xr.DataArray, labels: pd.DatetimeIndex
) -> xr.DataArray:
    """Linearly interpolate a 12-month climatology to each label's day centre."""
    if climatology.sizes["time"] != 12:
        raise ValueError(f"expected 12 months, got {climatology.sizes['time']}")
    stamps = pd.DatetimeIndex(climatology["time"].values)
    if not stamps.equals(_month_midpoints(range(stamps[0].year, stamps[0].year + 1))):
        raise ValueError(f"climatology is not stamped at month midpoints: {stamps}")
    # interp returns all-NaN when the source and target datetimes differ in
    # resolution (pandas builds date ranges in us, zarr times decode as ns).
    labels = labels.as_unit("ns")
    centres = labels - HALF_DAY
    # One year of padding either side covers the December-January wrap.
    midpoints = _month_midpoints(
        range(centres[0].year - 1, centres[-1].year + 2)
    ).as_unit("ns")
    monthly = xr.concat(
        [climatology.isel(time=t.month - 1, drop=True) for t in midpoints],
        dim=pd.Index(midpoints, name="time"),
    )
    daily = monthly.interp(time=centres.values, method="linear")
    return daily.assign_coords(time=labels.values)


def abrupt_ensemble_store_name(n: int) -> str:
    return f"abrupt4xCO2-ic_{n:04d}.zarr"


def build(storage_options: dict, template_store: str = TEMPLATE_STORE) -> xr.Dataset:
    """The Q-flux store on the time axis and grid of ``template_store``."""
    climatology = xr.open_zarr(QFLUX_CLIMATOLOGY, storage_options=storage_options)
    template = xr.open_zarr(template_store, storage_options=storage_options)
    for dim in ("grid_yt", "grid_xt"):
        if not np.array_equal(climatology[dim].values, template[dim].values):
            raise ValueError(f"{dim} of the climatology does not match the store")
    labels = pd.DatetimeIndex(template["time"].values)
    qflux = interpolate_climatology(climatology["Qflux"].load(), labels)
    if bool(qflux.isnull().any()):
        raise ValueError("interpolated Q-flux has NaNs")
    qflux = qflux.astype(np.float32).transpose("time", "grid_yt", "grid_xt")
    qflux.attrs = {
        **template[NAME].attrs,
        "long_name": "prescribed ocean Q-flux (re-derived at 4deg)",
        "source": QFLUX_CLIMATOLOGY,
    }
    ds = xr.Dataset({NAME: qflux})
    # A merged dataset takes its vertical coordinate from its first source,
    # which this store is, and the data side derives variables with it.
    for name in template.data_vars:
        if name.startswith(("ak_", "bk_")):
            ds[name] = template[name].load()
            ds[name].encoding = {}
    for name in ("time", "grid_yt", "grid_xt"):
        ds[name].attrs = template[name].attrs
        # The source stores' zarr v2 codecs do not carry over to a v3 write.
        ds[name].encoding = {}
    ds["time"].encoding = {
        "units": template["time"].encoding["units"],
        "calendar": template["time"].encoding["calendar"],
        "dtype": template["time"].encoding["dtype"],
    }
    ds[NAME].encoding = {
        "chunks": (min(360, ds.sizes["time"]), ds.sizes["grid_yt"], ds.sizes["grid_xt"])
    }
    ds.attrs["history"] = (
        f"{QFLUX_CLIMATOLOGY} interpolated to the day centres of the time axis of "
        f"{template_store} by configs/experiments/2026-06-26-fm/"
        "build_qflux_4deg_store.py."
    )
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build, do not write.")
    parser.add_argument(
        "--abrupt-ensemble",
        action="store_true",
        help="Write one store per abrupt-4xCO2 ensemble member, on its time axis.",
    )
    args = parser.parse_args()
    storage_options = _storage_options()
    if args.abrupt_ensemble:
        targets = [
            (
                f"{ABRUPT_ENSEMBLE_STORE_DIR}/{abrupt_ensemble_store_name(n)}",
                f"{ABRUPT_ENSEMBLE_OUTPUT_DIR}/{abrupt_ensemble_store_name(n)}",
            )
            for n in range(1, ABRUPT_ENSEMBLE_N_MEMBERS + 1)
        ]
    else:
        targets = [(TEMPLATE_STORE, OUTPUT_STORE)]
    for template_store, output_store in targets:
        ds = build(storage_options, template_store)
        print(ds)
        if args.dry_run:
            continue
        ds.to_zarr(output_store, mode="w-", storage_options=storage_options)
        print(f"Wrote {output_store}")


if __name__ == "__main__":
    main()
