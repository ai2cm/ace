"""Diagnose the form of the long-rollout blowup of the ERA5 4-deg residual model.

Reads the rollout output written by run_inference.sh (autoregressive_predictions.nc
and autoregressive_target.nc) plus the normalization statistics baked into the
checkpoint, and answers three questions:

  1. What variables are involved, and which goes out of sample first?
  2. What spatial/vertical form does the drift take early on?
  3. How far out of the training distribution does the state get, and when?

The key tool is the "out-of-sample" z-score: each predicted field is normalized
by the training-set mean and full-field std stored in the checkpoint, so z is
roughly N(0, 1) over the grid for in-distribution states. The model only ever saw
inputs with |z| of order a few during training, so a global-mean z drifting away
from 0, or max|z| climbing well past ~5, marks the state leaving the regime the
network was trained on.

Outputs go to output/diagnostics/: a set of PNG figures and a summary.md.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "output", "run")
CKPT = os.path.join(HERE, "checkpoint", "best_inference_ckpt.tar")
META = os.path.join(HERE, "data", "era5_4deg_blowup_slice.json")
OUT = os.path.join(HERE, "output", "diagnostics")

# z-score over the grid above which a state is clearly outside the training
# distribution (full-field std normalization makes in-sample |z| order a few).
OOS_THRESHOLD = 5.0
# Drift of the area-weighted global-mean z away from its day-0 value, in
# training-std units, that we treat as the onset of systematic drift.
DRIFT_THRESHOLD = 1.0

LEVEL_GROUPS = {
    "air_temperature": "air_temperature_{}",
    "specific_total_water": "specific_total_water_{}",
    "eastward_wind": "eastward_wind_{}",
    "northward_wind": "northward_wind_{}",
}


def load_norm_stats(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = ckpt["stepper"]["config"]["step"]["config"]["normalization"]["network"]
    return dict(norm["means"]), dict(norm["stds"])


def area_weights(lat):
    w = np.cos(np.deg2rad(lat.values))
    return w / w.mean()


def global_mean(da, w):
    # da dims: (time, lat, lon); w: (lat,)
    return (da * w[None, :, None]).mean(dim=("lat", "lon"))


def reconstruct_dates(n_time):
    start = "1979-01-01"
    if os.path.exists(META):
        start = json.load(open(META)).get("start", start)
    return np.array(
        [np.datetime64(start) + np.timedelta64(i + 1, "D") for i in range(n_time)]
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    gen = xr.open_dataset(os.path.join(RUN_DIR, "autoregressive_predictions.nc"))
    tgt = xr.open_dataset(os.path.join(RUN_DIR, "autoregressive_target.nc"))
    if "sample" in gen.sizes:
        gen = gen.isel(sample=0)
        tgt = tgt.isel(sample=0)
    means, stds = load_norm_stats(CKPT)
    w = area_weights(gen.lat)
    dates = reconstruct_dates(gen.sizes["time"])
    step = np.arange(gen.sizes["time"])

    # Every prognostic variable for which the checkpoint has normalization stats.
    # The target file from an inference (not evaluator) run only carries forcing
    # truth, so the target is used only for the physical-units comparison where
    # present; the out-of-sample z-score uses the training stats and needs no
    # target.
    variables = [v for v in gen.data_vars if v in means and v in stds]

    rows = []
    series = {}  # name -> dict of diagnostic time series
    for v in variables:
        g = gen[v].astype("float64")
        m, s = float(means[v]), float(stds[v])
        if not np.isfinite(s) or s == 0:
            continue
        zg = (g - m) / s  # normalized prediction

        gm_gen = global_mean(g, w).values
        gmz_gen = global_mean(zg, w).values  # state mean, in training-std units
        absz = np.abs(zg)
        maxz = absz.max(dim=("lat", "lon")).values
        frac_oos = (absz > OOS_THRESHOLD).mean(dim=("lat", "lon")).values

        if v in tgt:
            t = tgt[v].astype("float64")
            gm_tgt = global_mean(t, w).values
            tgt_maxz = float(np.abs((t - m) / s).max())
        else:
            gm_tgt = None
            tgt_maxz = np.nan

        # First NaN/inf step, if any.
        finite_per_step = np.isfinite(g.values).all(axis=(1, 2))
        nan_step = int(np.argmin(finite_per_step)) if not finite_per_step.all() else -1

        # Onset: first step where the global-mean state has drifted more than
        # DRIFT_THRESHOLD std units away from its (in-sample) day-0 value.
        drift = np.abs(gmz_gen - gmz_gen[0])
        over = drift > DRIFT_THRESHOLD
        onset = int(np.argmax(over)) if over.any() else -1

        series[v] = dict(
            gm_gen=gm_gen, gm_tgt=gm_tgt, gmz_gen=gmz_gen, maxz=maxz, frac_oos=frac_oos
        )
        rows.append(
            dict(
                var=v,
                onset_step=onset,
                onset_date=str(dates[onset])[:10] if onset >= 0 else "-",
                tgt_maxz=tgt_maxz,
                gen_maxz_final=float(maxz[-1]),
                gen_maxz_peak=float(np.nanmax(maxz)),
                final_gmz=float(gmz_gen[-1]),
                nan_step=nan_step,
            )
        )

    rows.sort(key=lambda r: (r["onset_step"] if r["onset_step"] >= 0 else 1 << 30))
    _write_summary(rows, dates, gen.sizes["time"])
    _plot_oos_onset(rows, series, step, dates)
    _plot_global_means(rows, series, step, dates)
    _plot_levels(gen, tgt, means, stds, w, step, dates)
    _plot_spatial(gen, means, stds, rows, dates)
    print(f"\nwrote diagnostics to {OUT}")


def _fmt_table(rows):
    hdr = (
        f"{'variable':<42}{'onset':>10}{'onset_date':>13}{'tgt|z|max':>11}"
        f"{'gen|z|peak':>12}{'final_mean_z':>14}{'nan@':>7}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        onset = r["onset_step"] if r["onset_step"] >= 0 else "-"
        nan = r["nan_step"] if r["nan_step"] >= 0 else "-"
        lines.append(
            f"{r['var']:<42}{str(onset):>10}{r['onset_date']:>13}"
            f"{r['tgt_maxz']:>11.2f}{r['gen_maxz_peak']:>12.1f}"
            f"{r['final_gmz']:>14.2f}{str(nan):>7}"
        )
    return "\n".join(lines)


def _write_summary(rows, dates, n_time):
    table = _fmt_table(rows)
    print(table)
    earliest = [r for r in rows if r["onset_step"] >= 0][:8]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# Blowup diagnostics\n\n")
        f.write(
            f"Rollout: {str(dates[0])[:10]} .. {str(dates[-1])[:10]} "
            f"({n_time} daily steps), single IC.\n\n"
        )
        f.write(
            "`onset` = first step where the area-weighted global-mean z-score "
            "(prediction normalized by training mean/std) has drifted more than "
            f"{DRIFT_THRESHOLD} std unit from its day-0 value. `|z|` values are "
            f"over the grid; OOS threshold |z|>{OOS_THRESHOLD}.\n\n"
        )
        f.write("## Variables ordered by onset of drift\n\n```\n")
        f.write(table + "\n```\n\n")
        if earliest:
            f.write("## Earliest variables to leave the training distribution\n\n")
            for r in earliest:
                ref = (
                    f" (target never exceeds {r['tgt_maxz']:.1f})"
                    if np.isfinite(r["tgt_maxz"])
                    else ""
                )
                f.write(
                    f"- **{r['var']}**: onset step {r['onset_step']} "
                    f"({r['onset_date']}), grid |z| peaks at {r['gen_maxz_peak']:.0f}"
                    f"{ref}.\n"
                )


def _plot_oos_onset(rows, series, step, dates):
    earliest = [r["var"] for r in rows if r["onset_step"] >= 0][:8]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for v in earliest:
        a.plot(step, series[v]["maxz"], label=v, lw=1.2)
        b.plot(step, series[v]["gmz_gen"], label=v, lw=1.2)
    a.axhline(OOS_THRESHOLD, color="k", ls="--", lw=0.8, label=f"|z|={OOS_THRESHOLD}")
    a.set_yscale("symlog")
    a.set_ylabel("max |z| over grid")
    a.set_title("Out-of-sample onset: grid-max normalized deviation")
    a.legend(fontsize=7, ncol=2)
    b.set_ylabel("global-mean z (std units)")
    b.set_xlabel("forward step (days)")
    b.set_title("Drift of the global-mean state, in training-std units")
    b.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oos_onset.png"), dpi=120)
    plt.close(fig)


def _plot_global_means(rows, series, step, dates):
    worst = sorted(rows, key=lambda r: -abs(r["final_gmz"]))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, r in zip(axes.flat, worst):
        v = r["var"]
        ax.plot(step, series[v]["gm_gen"], label="prediction", lw=1.3)
        if series[v]["gm_tgt"] is not None:
            ax.plot(step, series[v]["gm_tgt"], label="ERA5 target", lw=1.0, ls="--")
        ax.set_title(v, fontsize=9)
        ax.set_xlabel("step (days)")
        ax.legend(fontsize=7)
    fig.suptitle("Global-mean evolution: prediction vs target (physical units)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "global_means.png"), dpi=120)
    plt.close(fig)


def _plot_levels(gen, tgt, means, stds, w, step, dates):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (group, pat) in zip(axes.flat, LEVEL_GROUPS.items()):
        for lev in range(8):
            name = pat.format(lev)
            if name not in gen or name not in means:
                continue
            z = (gen[name].astype("float64") - means[name]) / stds[name]
            gmz = global_mean(z, w).values
            ax.plot(step, gmz, label=f"L{lev}", lw=1.0)
        ax.set_title(f"{group} (L0=top .. L7=near-surface)", fontsize=9)
        ax.set_ylabel("global-mean z")
        ax.set_xlabel("step (days)")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Vertical structure of the drift, per model level")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "level_structure.png"), dpi=120)
    plt.close(fig)


def _plot_spatial(gen, means, stds, rows, dates):
    # Focus on the earliest-diverging variable; show z maps over the rollout.
    earliest = [r for r in rows if r["onset_step"] >= 0]
    if not earliest:
        return
    v = earliest[0]["var"]
    z = (gen[v].astype("float64") - means[v]) / stds[v]
    n = gen.sizes["time"]
    onset = earliest[0]["onset_step"]
    times = sorted(set([0, max(onset, 1), n // 4, n // 2, 3 * n // 4, n - 1]))
    times = [t for t in times if 0 <= t < n][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    vmax = float(np.nanpercentile(np.abs(z.isel(time=times[-1]).values), 99)) or 5
    vmax = max(vmax, OOS_THRESHOLD)
    for ax, t in zip(axes.flat, times):
        im = ax.pcolormesh(
            gen.lon,
            gen.lat,
            z.isel(time=t).values,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"step {t} ({str(dates[t])[:10]})", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{v}: normalized deviation z over the grid (earliest to diverge)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spatial_z.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
