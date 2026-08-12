# Summary stats for a model's TC track verification, matching the schema
# already computed for st-flat/st-ou in
# crps_eval_results_stage2_st-flat-st-ou/tc_verification/tc_verification_summary.json:
# joins the known (reference) tracks against the model's rectified tracks on
# (track_id, time), computes SLP/wind MAE + Pearson correlation at matched
# points, plus track detection/confirmation counts.
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="TC track verification summary")
    parser.add_argument("--label", required=True)
    parser.add_argument("--known-csv", required=True, help="Reference (known) tracks.")
    parser.add_argument("--raw-csv", required=True, help="Model's raw (pre-rectification) detections.")
    parser.add_argument("--rectified-csv", required=True, help="Model's rectified tracks.")
    parser.add_argument("--outdir", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    known = pd.read_csv(args.known_csv, parse_dates=["time"])
    raw = pd.read_csv(args.raw_csv, parse_dates=["time"])
    rectified = pd.read_csv(args.rectified_csv, parse_dates=["time"])

    n_known_active = known["track_id"].nunique()
    n_tracks_detected = raw["track_id"].nunique()
    n_confirmed = rectified["track_id"].nunique()

    # mean_anchor_frac: per confirmed track, fraction of its matched points
    # that are "anchor" type (direct fine-center match) rather than
    # time-interpolated "interp" -- averaged across confirmed tracks.
    anchor_fracs = rectified.groupby("track_id")["point_type"].apply(
        lambda s: (s == "anchor").mean()
    )
    mean_anchor_frac = float(anchor_fracs.mean())

    merged = pd.merge(
        known[["track_id", "time", "slp", "wind"]],
        rectified[["track_id", "time", "slp", "wind"]],
        on=["track_id", "time"],
        suffixes=("_known", "_gen"),
    )

    slp_err = merged["slp_gen"] - merged["slp_known"]
    wind_err = merged["wind_gen"] - merged["wind_known"]

    summary = {
        "n_matched_points": int(len(merged)),
        "slp_mae_pa": float(slp_err.abs().mean()),
        "slp_corr": float(np.corrcoef(merged["slp_gen"], merged["slp_known"])[0, 1]),
        "wind_mae_ms": float(wind_err.abs().mean()),
        "wind_corr": float(np.corrcoef(merged["wind_gen"], merged["wind_known"])[0, 1]),
        "min_slp_gen_pa": float(merged["slp_gen"].min()),
        "max_wind_gen_ms": float(merged["wind_gen"].max()),
        "min_slp_known_pa": float(merged["slp_known"].min()),
        "max_wind_known_ms": float(merged["wind_known"].max()),
        "label": args.label,
        "n_tracks_detected": int(n_tracks_detected),
        "n_confirmed": int(n_confirmed),
        "n_known_active": int(n_known_active),
        "mean_anchor_frac": mean_anchor_frac,
    }

    with open(f"{args.outdir}/tc_verification_summary_{args.label}.json", "w") as f:
        json.dump([summary], f, indent=2)

    print(json.dumps(summary, indent=2))

    # ---- Track comparison map: known (black) vs matched rectified (red) ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for track_id, g in known.groupby("track_id"):
        ax.plot(g["lon"], g["lat"], color="black", lw=0.6, alpha=0.6)
    for track_id, g in rectified.groupby("track_id"):
        ax.plot(g["lon"], g["lat"], color="red", lw=0.6, alpha=0.6)
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude")
    ax.set_title(f"{args.label}: known tracks (black) vs. matched generated tracks (red)")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/tc_track_comparison_maps_{args.label}.png", dpi=150)
    plt.close(fig)

    # ---- Intensity scatter: known vs. generated SLP and wind at matched points ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(merged["slp_known"] / 100.0, merged["slp_gen"] / 100.0, s=4, alpha=0.3)
    lims = [merged[["slp_known", "slp_gen"]].min().min() / 100.0, merged[["slp_known", "slp_gen"]].max().max() / 100.0]
    axes[0].plot(lims, lims, color="gray", lw=0.8, ls="--")
    axes[0].set_xlabel("known SLP (mb)")
    axes[0].set_ylabel("generated SLP (mb)")
    axes[0].set_title(f"SLP (corr={summary['slp_corr']:.3f})")

    axes[1].scatter(merged["wind_known"], merged["wind_gen"], s=4, alpha=0.3)
    lims = [merged[["wind_known", "wind_gen"]].min().min(), merged[["wind_known", "wind_gen"]].max().max()]
    axes[1].plot(lims, lims, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("known wind (m/s)")
    axes[1].set_ylabel("generated wind (m/s)")
    axes[1].set_title(f"wind (corr={summary['wind_corr']:.3f})")

    fig.suptitle(f"{args.label}: TC intensity at matched points")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/tc_intensity_scatter_{args.label}.png", dpi=150)
    plt.close(fig)

    print(
        f"\nSaved tc_verification_summary_{args.label}.json, "
        f"tc_track_comparison_maps_{args.label}.png, tc_intensity_scatter_{args.label}.png"
    )


if __name__ == "__main__":
    main()
