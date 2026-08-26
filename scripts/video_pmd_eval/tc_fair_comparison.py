# Fair(er) TC track verification across models with different completeness.
#
# tc_summary.py's original scoring (Method A below) joins each model's own
# rectified_tracks.csv against the known-track reference on (track_id,
# time) -- an inner join, so a model only gets scored on the tracks/points
# it itself confirmed. That's a selection-bias trap: a model that only
# confirms its easiest storms reports error on a favorably-selected
# subset, making low-completeness models look artificially good on MAE.
#
# This script adds two corrections, computed side by side with the
# original so the effect of the correction is visible:
#   - Method B: restrict every model to the SAME set of tracks -- the
#     intersection of tracks confirmed by every model being compared.
#     Removes the "different subset" confound while still only scoring
#     matched points.
#   - Method C: score every known track's FULL 3-hourly timeseries. Any
#     known (track_id, time) a model didn't confirm is scored against a
#     fixed ambient no-storm value (SLP=101325 Pa, wind=0 m/s) instead of
#     being dropped. This is the strictest method and the one to trust for
#     "how good is this model at TC tracking, overall" -- it folds
#     detection failure into the intensity-accuracy number instead of
#     hiding it.
#   - signed_bias: mean(generated - known) on matched points only, for a
#     directional read (over- vs. under-intensifying) alongside the
#     unsigned MAE the three methods above report.
#
# Usage:
#   python tc_fair_comparison.py \
#     --known-csv known_tracks_2023_filtered_25km.csv \
#     --model st-flat=path/to/st-flat/rectified_tracks.csv \
#     --model hiro=path/to/hiro/rectified_tracks.csv \
#     [--model ... repeatable] \
#     --out-prefix /results/tc_fair
import argparse

import pandas as pd

AMBIENT_SLP_PA = 101325.0
AMBIENT_WIND_MS = 0.0


def _load_known(known_csv: str) -> pd.DataFrame:
    df = pd.read_csv(known_csv, parse_dates=["time"])
    return df[["track_id", "time", "slp", "wind"]].rename(
        columns={"slp": "slp_known", "wind": "wind_known"}
    )


def _load_rectified(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    return df[["track_id", "time", "slp", "wind"]].rename(
        columns={"slp": "slp_gen", "wind": "wind_gen"}
    )


def _mae_corr(
    merged: pd.DataFrame, known_col: str, gen_col: str
) -> tuple[float, float]:
    mae = float((merged[gen_col] - merged[known_col]).abs().mean())
    corr = float(merged[known_col].corr(merged[gen_col]))
    return mae, corr


def method_a(known: pd.DataFrame, rectified: pd.DataFrame) -> dict:
    """Original: inner join, each model scored only on what it confirmed."""
    merged = rectified.merge(known, on=["track_id", "time"], how="inner")
    slp_mae, slp_corr = _mae_corr(merged, "slp_known", "slp_gen")
    wind_mae, wind_corr = _mae_corr(merged, "wind_known", "wind_gen")
    confirmed_tracks = set(merged["track_id"].unique())
    return {
        "n_matched_points": len(merged),
        "n_confirmed_tracks": len(confirmed_tracks),
        "confirmed_tracks": confirmed_tracks,
        "slp_mae_mb": slp_mae / 100.0,
        "slp_corr": slp_corr,
        "wind_mae_ms": wind_mae,
        "wind_corr": wind_corr,
    }


def method_b(known: pd.DataFrame, rectified: pd.DataFrame, common_tracks: set) -> dict:
    """Common subset: restrict to tracks confirmed by every model compared."""
    known_common = known[known["track_id"].isin(common_tracks)]
    merged = rectified.merge(known_common, on=["track_id", "time"], how="inner")
    slp_mae, slp_corr = _mae_corr(merged, "slp_known", "slp_gen")
    wind_mae, wind_corr = _mae_corr(merged, "wind_known", "wind_gen")
    return {
        "n_matched_points": len(merged),
        "slp_mae_mb": slp_mae / 100.0,
        "slp_corr": slp_corr,
        "wind_mae_ms": wind_mae,
        "wind_corr": wind_corr,
    }


def method_c(known: pd.DataFrame, rectified: pd.DataFrame) -> dict:
    """All known tracks/timesteps, ambient fallback for any miss."""
    merged = known.merge(rectified, on=["track_id", "time"], how="left")
    missing = merged["slp_gen"].isna()
    merged.loc[missing, "slp_gen"] = AMBIENT_SLP_PA
    merged.loc[missing, "wind_gen"] = AMBIENT_WIND_MS
    slp_mae, slp_corr = _mae_corr(merged, "slp_known", "slp_gen")
    wind_mae, wind_corr = _mae_corr(merged, "wind_known", "wind_gen")
    return {
        "n_known_points": len(merged),
        "pct_missing": float(missing.mean() * 100.0),
        "slp_mae_mb": slp_mae / 100.0,
        "slp_corr": slp_corr,
        "wind_mae_ms": wind_mae,
        "wind_corr": wind_corr,
    }


def signed_bias(known: pd.DataFrame, rectified: pd.DataFrame) -> dict:
    """mean/median/std of (generated - known) on matched points only."""
    merged = rectified.merge(known, on=["track_id", "time"], how="inner")
    slp_diff = (merged["slp_gen"] - merged["slp_known"]) / 100.0
    wind_diff = merged["wind_gen"] - merged["wind_known"]
    return {
        "slp_signed_bias_mb": float(slp_diff.mean()),
        "slp_median_abs_err_mb": float(slp_diff.abs().median()),
        "slp_std_mb": float(slp_diff.std()),
        "wind_signed_bias_ms": float(wind_diff.mean()),
        "wind_median_abs_err_ms": float(wind_diff.abs().median()),
        "wind_std_ms": float(wind_diff.std()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-csv", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable. LABEL=path/to/rectified_tracks.csv",
    )
    parser.add_argument("--out-prefix", default=None, help="If set, write CSVs here.")
    return parser.parse_args()


def main():
    args = parse_args()
    known = _load_known(args.known_csv)
    n_known_tracks = known["track_id"].nunique()

    models: dict[str, pd.DataFrame] = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        models[label] = _load_rectified(path)

    a_rows, b_rows, c_rows, bias_rows = [], [], [], []

    a_results = {label: method_a(known, df) for label, df in models.items()}
    common_tracks = set.intersection(
        *(r["confirmed_tracks"] for r in a_results.values())
    )

    for label, df in models.items():
        a = a_results[label]
        a_rows.append(
            {
                "model": label,
                "confirmed": f"{a['n_confirmed_tracks']}/{n_known_tracks}",
                "slp_mae_mb": a["slp_mae_mb"],
                "slp_corr": a["slp_corr"],
                "wind_mae_ms": a["wind_mae_ms"],
                "wind_corr": a["wind_corr"],
            }
        )
        b = method_b(known, df, common_tracks)
        b_rows.append({"model": label, **b})
        c = method_c(known, df)
        c_rows.append({"model": label, **c})
        bias = signed_bias(known, df)
        bias_rows.append({"model": label, **bias})

    a_df = pd.DataFrame(a_rows)
    b_df = pd.DataFrame(b_rows)
    c_df = pd.DataFrame(c_rows)
    bias_df = pd.DataFrame(bias_rows)

    print(f"Known tracks: {n_known_tracks}")
    print(
        f"Common-subset intersection across all {len(models)} models: "
        f"{len(common_tracks)}/{n_known_tracks}"
    )
    print("\n=== Method A (original, each model's own confirmed set) ===")
    print(a_df.to_string(index=False))
    print("\n=== Method B (common subset) ===")
    print(b_df.to_string(index=False))
    print("\n=== Method C (all known tracks, ambient fallback) ===")
    print(c_df.to_string(index=False))
    print("\n=== Signed bias (matched points only) ===")
    print(bias_df.to_string(index=False))

    if args.out_prefix:
        a_df.to_csv(f"{args.out_prefix}_method_a.csv", index=False)
        b_df.to_csv(f"{args.out_prefix}_method_b.csv", index=False)
        c_df.to_csv(f"{args.out_prefix}_method_c.csv", index=False)
        bias_df.to_csv(f"{args.out_prefix}_signed_bias.csv", index=False)
        print(
            f"\nWrote {args.out_prefix}_method_{{a,b,c}}.csv, "
            f"{args.out_prefix}_signed_bias.csv"
        )


if __name__ == "__main__":
    main()
