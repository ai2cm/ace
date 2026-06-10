"""Probe 7: separate the secular global-mean bias from the seasonal cycle.

The probe 1+3 unperturbed ensembles showed gm drifts of ~1-2 milli-sigma/day
over 90-day windows, but such windows confound a true model bias with the
seasonal cycle of the global means. This probe removes the seasonal cycle by
differencing against ERA5: for each variable, compute the area-weighted
global mean of the saved rollout and of ERA5 (staged data file) on the same
dates, normalize both by the checkpoint's training std, and examine
(model - ERA5). The slope of that difference is the secular bias; the
seasonal cycle cancels.

Outputs to output/probe7_bias_vs_seasonal/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
PREDS_PATH = os.path.join(
    HERE, "output", "run_best_ckpt", "autoregressive_predictions.nc"
)
ERA5_PATH = os.path.join(HERE, "data", "era5_4deg_blowup_slice.nc")
CKPT = os.path.join(HERE, "checkpoint", "best_inference_ckpt.tar")
OUT = os.path.join(HERE, "output", "probe7_bias_vs_seasonal")

VARS = [
    "specific_total_water_0",
    "specific_total_water_1",
    "air_temperature_0",
    "air_temperature_7",
    "eastward_wind_0",
    "surface_temperature",
    "PRESsfc",
]
# Slope windows (steps): early, drift, late pre-onset
WINDOWS = {"0-500": (0, 500), "500-1500": (500, 1500), "1500-2100": (1500, 2100)}
ONSET_STEP = 2113


def load_norm_stats(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = ckpt["stepper"]["config"]["step"]["config"]["normalization"]["network"]
    return dict(norm["means"]), dict(norm["stds"])


def main():
    os.makedirs(OUT, exist_ok=True)
    means, stds = load_norm_stats(CKPT)

    preds = xr.open_dataset(PREDS_PATH)
    era5 = xr.open_dataset(ERA5_PATH)

    lat = preds["lat"].values
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    w2d = w[None, :, None]

    vt = pd.DatetimeIndex(np.asarray(preds["valid_time"].values).squeeze())
    n_time = len(vt)
    era5_times = pd.DatetimeIndex(era5["time"].values)
    era5_idx = era5_times.get_indexer(vt)
    if (era5_idx < 0).any():
        n_missing = int((era5_idx < 0).sum())
        raise ValueError(f"{n_missing} rollout dates missing from ERA5 slice")

    steps = np.arange(n_time)
    lines = [
        "# Probe 7: secular bias vs seasonal cycle",
        "",
        "gm(model) - gm(ERA5) on matching dates, in training-std units.",
        "Slopes in milli-sigma/day; the seasonal cycle cancels in the",
        "difference, so these slopes are the secular bias.",
        "",
        "| variable | "
        + " | ".join(f"bias {k}" for k in WINDOWS)
        + " | net drift @2100 (sigma) |",
        "|---|" + "---|" * (len(WINDOWS) + 1),
    ]
    fig, axes = plt.subplots(len(VARS), 1, figsize=(12, 2.2 * len(VARS)), sharex=True)
    for ax, v in zip(axes, VARS):
        s = float(stds[v])
        gm_model = (
            np.nansum(preds[v].values.squeeze() * w2d, axis=(1, 2))
            / (len(lat) * preds.sizes["lon"])
        ) / s
        e = era5[v].isel(time=era5_idx).values
        gm_era5 = (np.nansum(e * w2d, axis=(1, 2)) / (len(lat) * e.shape[2])) / s
        diff = gm_model - gm_era5

        row = [v]
        for k, (lo, hi) in WINDOWS.items():
            slope = np.polyfit(steps[lo:hi], diff[lo:hi], 1)[0]
            row.append(f"{slope * 1000:+.3f}")
        row.append(f"{diff[2100] - diff[0]:+.3f}")
        lines.append("| " + " | ".join(row) + " |")

        ax.plot(steps, diff, lw=0.7)
        ax.axvline(ONSET_STEP, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_ylabel(v, fontsize=7)
        ax.set_ylim(
            np.nanpercentile(diff[:ONSET_STEP], 0.5) - 0.2,
            np.nanpercentile(diff[:ONSET_STEP], 99.5) + 0.2,
        )
    axes[-1].set_xlabel("step (days)")
    fig.suptitle("gm(model) - gm(ERA5), sigma units (red = q0 OOS onset)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bias_timeseries.png"), dpi=150)

    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
