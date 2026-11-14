"""Append a constant global_mean_co2 to an E3SMv3 piControl zarr.

The E3SM output has no CO2 variable. The piControl run (compset WCYCL1850,
long name 1850SOI_EAM%CMIP6_...) sets co2vmr from CCSM_CO2_PPMV, which for
`^1850.+CMIP6` compsets is the CMIP6 1850 value of 284.317 ppmv
(driver-mct/cime_config/config_component_e3sm.xml on E3SM maint-3.0). The
variable matches the c96 datasets: dims (time,), float32, volume mixing ratio.

Only the new array is written; if the store has consolidated metadata it is
re-consolidated so the variable is visible to readers. Run it on a private
copy of the store (e.g. on weka), not on a store shared with other users.

Refuses to run if the store already has global_mean_co2.

Usage:
    python add_constant_co2.py /climate-default/<store>.zarr
    python add_constant_co2.py gs://.../<store>.zarr [--gcloud-token]
"""

import argparse
import subprocess

import numpy as np
import xarray as xr
import zarr

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


def _is_consolidated(store: str, storage_options: dict | None) -> bool:
    group = zarr.open_group(store, mode="r", storage_options=storage_options)
    return group.metadata.consolidated_metadata is not None


def add_constant_co2(store: str, storage_options: dict | None = None):
    consolidated = _is_consolidated(store, storage_options)
    ds = xr.open_zarr(store, storage_options=storage_options)
    if NAME in ds:
        raise ValueError(f"{store} already has {NAME}")
    n_time = ds.sizes["time"]
    co2 = xr.DataArray(
        np.full(n_time, CO2_PPMV * 1e-6, dtype=np.float32),
        dims=["time"],
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
    co2.to_dataset(name=NAME).to_zarr(
        store,
        mode="a",
        storage_options=storage_options,
        encoding={NAME: {"chunks": (n_time,)}},
        consolidated=consolidated,
    )

    # Check what a reader sees, through the same (consolidated) metadata.
    written = xr.open_zarr(store, storage_options=storage_options)
    if set(written.variables) != set(ds.variables) | {NAME}:
        raise RuntimeError(
            f"Variables of {store} changed unexpectedly: "
            f"{sorted(set(written.variables) ^ set(ds.variables))}"
        )
    values = written[NAME].values
    if values.shape != (n_time,) or not np.all(values == np.float32(CO2_PPMV * 1e-6)):
        raise RuntimeError(f"{NAME} in {store} was not written correctly")
    print(
        f"Wrote {NAME} = {CO2_PPMV * 1e-6:.6e} ({n_time} steps) to {store} "
        f"(consolidated metadata: {consolidated})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store")
    parser.add_argument(
        "--gcloud-token",
        action="store_true",
        help="Authenticate with `gcloud auth print-access-token`.",
    )
    args = parser.parse_args()
    add_constant_co2(args.store, _storage_options(args.gcloud_token))


if __name__ == "__main__":
    main()
