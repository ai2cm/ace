"""Probe 5: does specific total water approach saturation during the drift/blowup?

Computes RH_i = q_i / qsat(T_i, p_i) for each model level over the saved
run_best_ckpt rollout, with level mid pressures from the checkpoint's ak/bk
interface coefficients and predicted PRESsfc, and Bolton saturation vapor
pressure (the same formula as the qsat-scaling feature; over liquid water, so
upper-level "RH" values are conservative/underestimated relative to ice).

Answers whether a saturation-bound corrector (q_i <= alpha * qsat) would bind
early in the drift, or only after temperature has drifted enough to raise qsat
in tandem. Outputs to output/probe5_saturation/: per-level time series plots
and a summary table.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
PREDS_PATH = os.path.join(
    HERE, "output", "run_best_ckpt", "autoregressive_predictions.nc"
)
CKPT = os.path.join(HERE, "checkpoint", "best_inference_ckpt.tar")
OUT = os.path.join(HERE, "output", "probe5_saturation")

ONSET_STEP = 2113  # first OOS onset (specific_total_water_0) in this rollout
N_LEVELS = 8


def bolton_saturation_vapor_pressure(temperature_k):
    """Bolton (1980) saturation vapor pressure over liquid water, in Pa."""
    t_c = temperature_k - 273.15
    return 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))


def saturation_specific_humidity(temperature_k, pressure_pa):
    es = bolton_saturation_vapor_pressure(temperature_k)
    # Guard: at extreme blowup temperatures es can exceed p; qsat is then
    # meaningless, mark as NaN rather than returning negative values.
    denom = pressure_pa - 0.378 * es
    qsat = 0.622 * es / denom
    qsat = np.where((denom <= 0) | (es >= pressure_pa), np.nan, qsat)
    return qsat


def load_ak_bk(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vc = ckpt["stepper"]["dataset_info"]["vertical_coordinate"]
    return vc["ak"].numpy(), vc["bk"].numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    ak, bk = load_ak_bk(CKPT)
    print(f"ak interfaces: {ak}")
    print(f"bk interfaces: {bk}")

    ds = xr.open_dataset(PREDS_PATH)
    lat = ds["lat"].values
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    w2d = w[None, :, None]
    n_time = ds.sizes["time"]
    steps = np.arange(n_time)

    ps = ds["PRESsfc"].values.squeeze()  # (time, lat, lon)

    stats = {}
    print(f"\n{'lvl':>3} {'p_mid(1979) hPa':>15}")
    for i in range(N_LEVELS):
        p_top = ak[i] + bk[i] * ps
        p_bot = ak[i + 1] + bk[i + 1] * ps
        p_mid = 0.5 * (p_top + p_bot)
        print(f"{i:>3} {np.mean(p_mid[:30]) / 100:>15.1f}")

        t = ds[f"air_temperature_{i}"].values.squeeze()
        q = ds[f"specific_total_water_{i}"].values.squeeze()
        qsat = saturation_specific_humidity(t, p_mid)
        rh = q / qsat

        stats[i] = {
            "mean": np.nanmean(rh * w2d, axis=(1, 2)),
            "max": np.nanmax(rh, axis=(1, 2)),
            "frac_gt_1": np.nanmean(rh > 1.0, axis=(1, 2)),
            "frac_gt_1.2": np.nanmean(rh > 1.2, axis=(1, 2)),
            "nan_frac": np.mean(~np.isfinite(rh), axis=(1, 2)),
        }
        del t, q, qsat, rh

    # Summary table: pre-onset drift phase vs early cascade vs first month
    def window_stats(s, lo, hi):
        return (
            np.nanmean(s["mean"][lo:hi]),
            np.nanmax(s["max"][lo:hi]),
            np.nanmean(s["frac_gt_1"][lo:hi]),
        )

    windows = {
        "first 90d (in-sample)": (0, 90),
        "drift (1500-2100)": (1500, 2100),
        "cascade (2113-2400)": (2113, 2400),
        "deep blowup (2400-end)": (2400, n_time),
    }
    lines = [
        "# Probe 5: saturation check on run_best_ckpt rollout",
        "",
        "RH = specific_total_water / qsat(T, p_mid), Bolton es over liquid water.",
        f"Onset step (q0 OOS): {ONSET_STEP}.",
        "",
    ]
    for name, (lo, hi) in windows.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| lvl | mean RH | max RH | area frac RH>1 |")
        lines.append("|---|---|---|---|")
        for i in range(N_LEVELS):
            m, mx, fr = window_stats(stats[i], lo, hi)
            lines.append(f"| {i} | {m:.3f} | {mx:.2f} | {fr:.4f} |")
        lines.append("")
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)

    # Plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i in range(N_LEVELS):
        axes[0].plot(steps, stats[i]["mean"], lw=0.8, label=f"lvl {i}")
        axes[1].plot(steps, stats[i]["max"], lw=0.8, label=f"lvl {i}")
        axes[2].plot(steps, stats[i]["frac_gt_1"], lw=0.8, label=f"lvl {i}")
    for ax, title in zip(
        axes, ["global-mean RH", "grid-max RH", "area fraction RH > 1"]
    ):
        ax.axvline(ONSET_STEP, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(title)
        ax.legend(ncol=4, fontsize=8)
    axes[1].set_yscale("log")
    axes[2].set_xlabel("step (days)")
    fig.suptitle("Probe 5: saturation ratio per level (red line = q0 OOS onset)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rh_timeseries.png"), dpi=150)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
