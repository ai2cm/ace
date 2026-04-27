"""Inter-model dispersion of mean, std, and d1_std (T1–T3).

Reads the per-dataset stats produced by ``compute_stats.py`` and, for each
``(variable, plev_index)`` group, computes how much the per-dataset value
of each statistic varies across models. We collapse to one value per
``label`` (i.e. ``source_id.p``) by averaging across that label's
realizations and experiments, so each model contributes one point.

Outputs into ``outputs/``:

- ``{stat}_dispersion.csv`` — one row per (variable, plev_index) with
  median / IQR / max-over-median / coefficient-of-variation across
  models, plus the count of models contributing.
- ``dispersion_summary.md`` — human-readable rollup grouped by category
  (3D state, 2D state, fluxes, forcings, statics) flagging variables
  where dispersion is large enough to motivate per-dataset scales.

The "interesting" threshold is CoV > 0.30 (30%) for std/d1_std and
abs(mean) / median(std) > 0.10 for mean offsets — both arbitrary but
calibrated against ACE-style training experience where larger spreads
push you toward per-dataset normalization.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("normalization.dispersion")

# Variables whose dispersion is interesting. We keep "ta" out of stats
# directly (it's not in the dataset) but ta_derived_layer_* should be
# treated as state.
_STATE_3D = ["ua", "va", "hus", "zg"]
_STATE_2D_DERIVED_T = [f"ta_derived_layer_{i}" for i in range(7)]
_STATE_2D = ["tas", "huss", "psl", "pr"]
_FLUXES = [
    "rsdt",
    "rsut",
    "rlut",
    "rsds",
    "rsus",
    "rlds",
    "rlus",
    "hfss",
    "hfls",
]
_SURFACE_WIND = ["sfcWind", "uas", "vas"]
_FORCINGS = ["ts", "siconc"]
_STATICS = ["sftlf", "orog"]

_CATEGORY = {}
for v in _STATE_3D:
    _CATEGORY[v] = "3D state"
for v in _STATE_2D_DERIVED_T:
    _CATEGORY[v] = "derived layer T"
for v in _STATE_2D:
    _CATEGORY[v] = "2D state"
for v in _FLUXES:
    _CATEGORY[v] = "flux"
for v in _SURFACE_WIND:
    _CATEGORY[v] = "surface wind"
for v in _FORCINGS:
    _CATEGORY[v] = "forcing"
for v in _STATICS:
    _CATEGORY[v] = "static"


def _collapse_to_per_label(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Average a stat across realizations/experiments for each label.

    A model's std (or mean, or d1_std) is reasonably stable across
    realizations and slightly less so across historical/ssp585 — but for
    inter-*model* dispersion the per-realization scatter is noise we
    want to suppress. Mean over (label, variable, plev_index) gives one
    representative number per model.
    """
    df = df[df[stat].notna()].copy()
    if df.empty:
        return df
    return (
        df.groupby(["label", "variable", "plev_index"], dropna=False)[stat]
        .mean()
        .reset_index()
    )


def _dispersion_table(per_label: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Compute dispersion summary across models.

    For each (variable, plev_index), we report:

    - ``n_models``: number of labels with this stat present.
    - ``median``: median of the per-model stat.
    - ``iqr``: 75th - 25th percentile.
    - ``min``, ``max``.
    - ``max_over_median``: max / median (robust dynamic range).
    - ``cov``: std / mean (coefficient of variation; robust-ish since
      most values are positive). For ``mean`` of variables centred near
      zero, CoV is unreliable, so we also report ``iqr_over_median_std``
      (see ``_dispersion_table_mean``).
    """
    rows = []
    for (var, plev), g in per_label.groupby(["variable", "plev_index"], dropna=False):
        vals = g[stat].to_numpy()
        n = len(vals)
        if n < 2:
            continue
        med = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        mn = float(vals.min())
        mx = float(vals.max())
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "variable": var,
                "plev_index": plev,
                "category": _CATEGORY.get(var, "other"),
                "n_models": n,
                "median": med,
                "iqr": float(q3 - q1),
                "min": mn,
                "max": mx,
                "max_over_median": mx / med if med != 0 else np.nan,
                "cov": sd / abs(mean) if mean != 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["category", "variable", "plev_index"], na_position="last"
        ).reset_index(drop=True)
    return out


def _dispersion_table_mean(
    per_label_mean: pd.DataFrame, per_label_std: pd.DataFrame
) -> pd.DataFrame:
    """Dispersion of the mean, using std-units to make it comparable.

    A coefficient of variation of the mean is useless for variables
    centred at zero (ua, va anomalies). Instead we report the spread of
    the per-model means in units of the median per-model std — i.e.
    "how many sigmas of within-model variability separate the model
    means?". > 0.5 is large.
    """
    std_med = (
        per_label_std.groupby(["variable", "plev_index"], dropna=False)["std"]
        .median()
        .rename("median_std")
        .reset_index()
    )
    rows = []
    for (var, plev), g in per_label_mean.groupby(
        ["variable", "plev_index"], dropna=False
    ):
        vals = g["mean"].to_numpy()
        n = len(vals)
        if n < 2:
            continue
        if pd.isna(plev):
            ms_mask = std_med["variable"].eq(var) & std_med["plev_index"].isna()
        else:
            ms_mask = std_med["variable"].eq(var) & std_med["plev_index"].eq(plev)
        ms = std_med.loc[ms_mask, "median_std"]
        med_std = float(ms.iloc[0]) if len(ms) else np.nan
        med = float(np.median(vals))
        mn = float(vals.min())
        mx = float(vals.max())
        mean_spread = float(vals.std(ddof=1))
        rows.append(
            {
                "variable": var,
                "plev_index": plev,
                "category": _CATEGORY.get(var, "other"),
                "n_models": n,
                "median_mean": med,
                "min_mean": mn,
                "max_mean": mx,
                "spread_mean": mean_spread,
                "median_std": med_std,
                "spread_in_sigmas": (
                    mean_spread / med_std if med_std and med_std > 0 else np.nan
                ),
                "range_in_sigmas": (
                    (mx - mn) / med_std if med_std and med_std > 0 else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["category", "variable", "plev_index"], na_position="last"
        ).reset_index(drop=True)
    return out


def _format_var_for_summary(var: str, plev: float | None) -> str:
    if pd.isna(plev) or plev is None:
        return var
    return f"{var} (plev_index={int(plev)})"


def _summarize_for_humans(
    std_disp: pd.DataFrame,
    d1_disp: pd.DataFrame,
    mean_disp: pd.DataFrame,
    out_path: Path,
    cov_threshold: float = 0.30,
    sigma_threshold: float = 0.5,
) -> None:
    lines = ["# Inter-model dispersion summary", ""]
    n_models = std_disp["n_models"].max() if not std_disp.empty else "?"
    lines.append(
        f"Per-variable dispersion of std, d1_std, and mean across "
        f"{n_models} models (one value per model, averaged over "
        "realizations and experiments)."
    )
    lines.append("")
    lines.append(
        f"**Flags**: CoV > {cov_threshold:.0%} on std/d1_std, or model-mean spread > "
        f"{sigma_threshold:.1f}σ of within-model variability — these are the variables "
        "where shared scales lose meaningful conditioning vs per-dataset."
    )
    lines.append("")

    def _flagged(df: pd.DataFrame, col: str, thresh: float) -> pd.DataFrame:
        return df[df[col] > thresh].sort_values(col, ascending=False)

    lines.append("## std (cell-time)")
    lines.append("")
    flagged = _flagged(std_disp, "cov", cov_threshold)
    if flagged.empty:
        lines.append(f"_No variables flagged at CoV > {cov_threshold:.0%}._")
    else:
        lines.append("| variable | n | median | max/median | CoV |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in flagged.iterrows():
            lines.append(
                f"| {_format_var_for_summary(r['variable'], r['plev_index'])} "
                f"| {int(r['n_models'])} "
                f"| {r['median']:.3g} "
                f"| {r['max_over_median']:.2f} "
                f"| {r['cov']:.0%} |"
            )
    lines.append("")

    lines.append("## d1_std (one-step finite difference)")
    lines.append("")
    flagged = _flagged(d1_disp, "cov", cov_threshold)
    if flagged.empty:
        lines.append(f"_No variables flagged at CoV > {cov_threshold:.0%}._")
    else:
        lines.append("| variable | n | median | max/median | CoV |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in flagged.iterrows():
            lines.append(
                f"| {_format_var_for_summary(r['variable'], r['plev_index'])} "
                f"| {int(r['n_models'])} "
                f"| {r['median']:.3g} "
                f"| {r['max_over_median']:.2f} "
                f"| {r['cov']:.0%} |"
            )
    lines.append("")

    lines.append("## mean (in std-units)")
    lines.append("")
    flagged = mean_disp[mean_disp["spread_in_sigmas"] > sigma_threshold].sort_values(
        "spread_in_sigmas", ascending=False
    )
    if flagged.empty:
        lines.append(
            f"_No variables flagged at model-mean spread > {sigma_threshold}σ._"
        )
    else:
        lines.append("| variable | n | median mean | spread (σ) | range (σ) |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in flagged.iterrows():
            lines.append(
                f"| {_format_var_for_summary(r['variable'], r['plev_index'])} "
                f"| {int(r['n_models'])} "
                f"| {r['median_mean']:.3g} "
                f"| {r['spread_in_sigmas']:.2f} "
                f"| {r['range_in_sigmas']:.2f} |"
            )
    lines.append("")

    lines.append("## All variables — std dispersion (full table)")
    lines.append("")
    lines.append("| variable | n | median std | max/median | CoV |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in std_disp.iterrows():
        lines.append(
            f"| {_format_var_for_summary(r['variable'], r['plev_index'])} "
            f"| {int(r['n_models'])} "
            f"| {r['median']:.3g} "
            f"| {r['max_over_median']:.2f} "
            f"| {r['cov']:.0%} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines))
    LOG.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats",
        default="data/cmip6-daily-pilot/v0/stats.csv",
        help="Path to stats.csv from compute_stats.py",
    )
    parser.add_argument(
        "--out-dir",
        default="normalization/outputs",
        help="Directory to write dispersion tables and summary",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = pd.read_csv(args.stats)
    LOG.info("Loaded %d stat rows", len(stats))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_label_std = _collapse_to_per_label(stats, "std")
    per_label_d1 = _collapse_to_per_label(stats, "d1_std")
    per_label_mean = _collapse_to_per_label(stats, "mean")

    std_disp = _dispersion_table(per_label_std, "std")
    d1_disp = _dispersion_table(per_label_d1, "d1_std")
    mean_disp = _dispersion_table_mean(per_label_mean, per_label_std)

    std_disp.to_csv(out_dir / "std_dispersion.csv", index=False)
    d1_disp.to_csv(out_dir / "d1_std_dispersion.csv", index=False)
    mean_disp.to_csv(out_dir / "mean_dispersion.csv", index=False)
    LOG.info(
        "Wrote dispersion CSVs (std=%d rows, d1_std=%d rows, mean=%d rows)",
        len(std_disp),
        len(d1_disp),
        len(mean_disp),
    )

    _summarize_for_humans(
        std_disp, d1_disp, mean_disp, out_dir / "dispersion_summary.md"
    )


if __name__ == "__main__":
    main()
