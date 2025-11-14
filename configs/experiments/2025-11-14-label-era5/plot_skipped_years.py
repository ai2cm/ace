import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

VAR = "specific_total_water_0"
PAD_BEFORE = 2  # timesteps before start
PAD_AFTER = 3  # timesteps after stop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)

    out_filename = os.path.basename(args.config_path).replace(
        ".yaml", "-skipped_years.png"
    )

    # Gather ERA5 subsets and derive skipped (gap) periods
    era5_subsets = [
        d["subset"]
        for d in cfg["train_loader"]["dataset"]["concat"]
        if "era5" in d.get("labels", [])
    ]
    skipped_periods = []
    prev = era5_subsets[0]
    for sub in era5_subsets[1:]:
        skipped_periods.append((prev["stop_time"], sub["start_time"]))
        prev = sub

    ds = xr.open_zarr(
        "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
    )
    time_index = ds.indexes["time"]  # pandas.DatetimeIndex
    weights = np.cos(np.deg2rad(ds.latitude))

    nrows = len(skipped_periods)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(6, 3.2 * nrows), squeeze=False
    )

    for r, (start_str, stop_str) in enumerate(skipped_periods):
        # find padded slice indices around inspection window
        start_i = time_index.get_loc(np.datetime64(start_str))
        stop_i = time_index.get_loc(np.datetime64(stop_str))
        lo = max(0, start_i - PAD_BEFORE)
        hi = min(len(time_index), stop_i + PAD_AFTER)

        ds_win = ds.isel(time=slice(lo, hi))

        # metrics within the window (compute only what we plot)
        gm = ds_win[VAR].weighted(weights).mean(("latitude", "longitude"))
        dvar = ds_win[VAR].diff("time")
        rms = np.sqrt((dvar**2).mean(("latitude", "longitude")))

        t_gm = gm["time"].values
        t_rms = rms["time"].values  # note: diff drops first step

        ax_gm = axes[r][0]

        # plot global mean
        ax_gm.plot(t_gm, gm.values, lw=1.2)
        ax_gm.set_ylabel("Global mean")
        ax_gm.set_title(f"Gap {start_str} → {stop_str}\nfor {VAR}")
        # shade inspection period
        s_shade = np.datetime64(start_str)
        e_shade = np.datetime64(stop_str)
        ax_gm.axvspan(s_shade, e_shade, alpha=0.25)

        # tidy x labels: only bottom row gets labels
        if r == nrows - 1:
            ax_gm.set_xlabel("Time")
        else:
            ax_gm.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(out_filename)
