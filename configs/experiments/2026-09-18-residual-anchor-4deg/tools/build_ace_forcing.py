#!/usr/bin/env python
r"""Build an ocean-forcing zarr whose ten atmosphere-supplied fields are ACE's own
5-day means from a coupled rollout, on the CM4 ocean store's full time axis
(CM4 values wherever the rollout has none).

Runs on Beaker with the coupled rollout's result dataset mounted at /coupled and
weka at /climate-default; writes /results/ace_forcing.zarr. The coupled writer
block-averaged the 20 atmosphere steps of each ocean step; here block k is
assigned the ocean store time that follows the rollout's k-th initial time,
which is where the ocean stepper's next-step forcing convention reads it.

    python build_ace_forcing.py \\
        --coupled /coupled/atmosphere/autoregressive_predictions.nc \\
        --store /climate-default/2026-07-22-cm4-picontrol-4deg-coupled-ocean.zarr \\
        --out /results/ace_forcing.zarr
"""

import argparse

import numpy as np
import xarray as xr

FIELDS = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "eastward_surface_wind_stress",
    "northward_surface_wind_stress",
    "total_frozen_precipitation_rate",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    ace = xr.open_dataset(a.coupled, decode_timedelta=True)
    store = xr.open_zarr(a.store)
    missing = [f for f in FIELDS if f not in ace or f not in store]
    if missing:
        raise SystemExit(
            f"fields missing from the coupled file or the store: {missing}"
        )
    ace = ace.isel(sample=a.sample) if "sample" in ace.dims else ace
    # block-mean labels sit inside the block; the IC is the last store time
    # at or before the first label
    ace_times = ace["valid_time"].values if "valid_time" in ace else ace["time"].values
    store_times = store["time"].values
    ic = int(np.searchsorted(store_times, ace_times[0], side="right") - 1)
    n = ace.sizes["time"]
    target = store_times[ic + 1 : ic + 1 + n]
    if len(target) != n:
        raise SystemExit(
            f"rollout has {n} blocks; the store has {len(target)} times after the IC"
        )
    print(f"IC store index {ic} ({store_times[ic]}); {n} blocks")
    print(f"  -> store times {target[0]} .. {target[-1]}")
    out = store[FIELDS].load()
    for f in FIELDS:
        vals = ace[f].values  # (time, lat, lon)
        out[f].values[ic + 1 : ic + 1 + n] = vals.astype(out[f].dtype)
        cm4 = np.nanmean(store[f].values[ic + 1 : ic + 1 + n])
        print(f"  {f}: ACE mean {np.nanmean(vals):.4g} vs CM4 mean {cm4:.4g}")
    out.attrs["source"] = (
        f"ACE fluxes from {a.coupled} over {n} ocean steps from {store_times[ic]}; "
        f"CM4 ({a.store}) elsewhere"
    )
    out.to_zarr(a.out, mode="w")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
