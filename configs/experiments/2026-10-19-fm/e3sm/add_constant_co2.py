"""Append a constant global_mean_co2 to the daily E3SMv3 piControl zarr.

The E3SM output has no CO2 variable. The piControl run (compset WCYCL1850,
long name 1850SOI_EAM%CMIP6_...) sets co2vmr from CCSM_CO2_PPMV, which for
`^1850.+CMIP6` compsets is the CMIP6 1850 value of 284.317 ppmv
(driver-mct/cime_config/config_component_e3sm.xml on E3SM maint-3.0). The
variable matches the c96 datasets: dims (time,), float32, volume mixing ratio.

Refuses to run if the store already has global_mean_co2.

Usage:
    python add_constant_co2.py gs://.../<store>.zarr [--gcloud-token]
"""

import argparse
import subprocess

import numpy as np
import xarray as xr

NAME = "global_mean_co2"
CO2_PPMV = 284.317


def _storage_options(gcloud_token: bool) -> dict | None:
    if not gcloud_token:
        return None
    from google.oauth2.credentials import Credentials

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"token": Credentials(token)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store")
    parser.add_argument(
        "--gcloud-token",
        action="store_true",
        help="Authenticate with `gcloud auth print-access-token`.",
    )
    args = parser.parse_args()
    storage_options = _storage_options(args.gcloud_token)

    ds = xr.open_zarr(args.store, storage_options=storage_options)
    if NAME in ds:
        raise ValueError(f"{args.store} already has {NAME}")
    co2 = xr.DataArray(
        np.full(ds.sizes["time"], CO2_PPMV * 1e-6, dtype=np.float32),
        dims=["time"],
        coords={"time": ds.time},
        attrs={
            "long_name": "global mean carbon dioxide concentration",
            "units": "volume mixing ratio",
            "cell_methods": "time: point",
            "description": (
                f"Constant CMIP6 1850 value ({CO2_PPMV} ppmv) used by the "
                "E3SMv3 piControl run; added by add_constant_co2.py."
            ),
        },
    )
    co2.to_dataset(name=NAME).drop_vars("time").to_zarr(
        args.store,
        mode="a",
        storage_options=storage_options,
        encoding={NAME: {"chunks": (ds.sizes["time"],)}},
    )
    print(f"Wrote {NAME} = {CO2_PPMV * 1e-6:.6e} to {args.store}")


if __name__ == "__main__":
    main()
