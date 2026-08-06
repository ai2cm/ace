#!/usr/bin/env python3
"""Diagnostic figure for lead-1 Nino3.4 forecasts across all OOS years.

Per-year ACC at lead 1 is computed from only 12 ICs, so it is dominated by how
much Nino3.4 variance that year happens to contain. This script plots the raw
lead-1 pairs so the low- and negative-ACC years can be inspected directly.

Example::

    python plot_lead1_diagnostics.py --year-start 231 --year-end 250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEAD = 1


def _init_sort_key(init_time: str) -> tuple[int, int, int]:
    """Sortable (year, month, day) from a cftime string like '0230-12-29 ...'.

    ``pandas.to_datetime`` cannot represent these model years, so parse the
    fixed-width date fields directly.
    """
    return (int(init_time[:4]), int(init_time[5:7]), int(init_time[8:10]))


def load_lead1(analysis_dir: Path, year_start: int, year_end: int) -> pd.DataFrame:
    frames = []
    for year in range(year_start, year_end + 1):
        path = analysis_dir / f"yr{year:04d}" / "nino_ar_sst_forecasts.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame.lead_month == LEAD].copy()
        frame["job_year"] = year
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No per-year forecast CSVs under {analysis_dir}")
    data = pd.concat(frames, ignore_index=True)
    keys = data.init_time.map(_init_sort_key)
    data["init_year"] = [k[0] for k in keys]
    data["init_month"] = [k[1] for k in keys]
    return data.sort_values(["init_year", "init_month"]).reset_index(drop=True)


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% confidence interval for a correlation via the Fisher z transform."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    # 1.96 for alpha=0.05; scipy is not a dependency of this analysis dir.
    half_width = 1.959963985 * (1.0 / np.sqrt(n - 3))
    return (float(np.tanh(z - half_width)), float(np.tanh(z + half_width)))


def per_year_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in data.groupby("job_year"):
        rmse = float(np.sqrt(((group.prediction - group.target) ** 2).mean()))
        acc = float(group.prediction.corr(group.target))
        lo, hi = _fisher_ci(acc, len(group))
        rows.append(
            {
                "job_year": year,
                "n": len(group),
                "acc": acc,
                "acc_lo": lo,
                "acc_hi": hi,
                "rmse": rmse,
                "bias": float((group.prediction - group.target).mean()),
                "target_std": float(group.target.std()),
                "pred_std": float(group.prediction.std()),
                "snr": float(group.target.std()) / rmse if rmse > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    data: pd.DataFrame,
    stats: pd.DataFrame,
    output_path: Path,
    highlight: tuple[int, ...],
) -> None:
    pooled_acc = float(data.prediction.corr(data.target))
    pooled_rmse = float(np.sqrt(((data.prediction - data.target) ** 2).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # (a) pooled scatter, highlighting the suspicious years
    ax = axes[0, 0]
    other = data[~data.job_year.isin(highlight)]
    ax.scatter(
        other.target, other.prediction, s=26, c="0.6", label="other years", zorder=2
    )
    colors = ["#d62728", "#ff7f0e", "#9467bd"]
    for color, year in zip(colors, highlight):
        sel = data[data.job_year == year]
        ax.scatter(
            sel.target,
            sel.prediction,
            s=70,
            c=color,
            edgecolor="k",
            linewidth=0.5,
            label=f"yr{year:04d}",
            zorder=3,
        )
    lim = [
        min(data.target.min(), data.prediction.min()) - 0.2,
        max(data.target.max(), data.prediction.max()) + 0.2,
    ]
    ax.plot(lim, lim, "k--", linewidth=1, zorder=1, label="1:1")
    ax.set(
        xlim=lim,
        ylim=lim,
        xlabel="Target Nino3.4 anomaly (K)",
        ylabel="Predicted Nino3.4 anomaly (K)",
        title=(
            f"(a) All {len(data)} lead-1 forecasts\n"
            f"pooled ACC={pooled_acc:.3f}, RMSE={pooled_rmse:.2f} K"
        ),
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) lead-1 time series in IC order, shading the highlighted years
    ax = axes[0, 1]
    x = np.arange(len(data))
    ax.plot(x, data.target, "-o", ms=3, lw=1.2, color="C1", label="target")
    ax.plot(x, data.prediction, "-o", ms=3, lw=1.2, color="C0", label="prediction")
    for color, year in zip(colors, highlight):
        idx = x[data.job_year.values == year]
        if len(idx):
            ax.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color=color, alpha=0.15)
            ax.text(
                idx.mean(),
                lim[1] * 0.94,
                f"yr{year:04d}",
                ha="center",
                fontsize=9,
                color=color,
                weight="bold",
            )
    ax.axhline(0, color="k", lw=0.8)
    year_starts = [
        (x[data.job_year.values == y][0], y)
        for y in sorted(data.job_year.unique())
        if len(x[data.job_year.values == y])
    ]
    ax.set_xticks([p for p, _ in year_starts][::2])
    ax.set_xticklabels([f"{y:04d}" for _, y in year_starts][::2], rotation=45)
    ax.set(
        xlabel="Initialization (grouped by model year)",
        ylabel="Nino3.4 anomaly (K)",
        title=(
            "(b) Lead-1 prediction tracks target everywhere;\n"
            "low-ACC years are simply flat"
        ),
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) why per-year ACC collapses: it follows signal-to-noise, not error
    ax = axes[1, 0]
    ax.errorbar(
        stats.target_std,
        stats.acc,
        yerr=[stats.acc - stats.acc_lo, stats.acc_hi - stats.acc],
        fmt="none",
        ecolor="0.7",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    sc = ax.scatter(
        stats.target_std,
        stats.acc,
        s=90,
        c=stats.rmse,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.5,
        zorder=2,
    )
    for _, row in stats.iterrows():
        ax.annotate(
            f"{int(row.job_year):04d}",
            (row.target_std, row.acc),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(
        pooled_acc,
        color="C3",
        ls="--",
        lw=1.2,
        label=f"pooled ACC over all {len(data)} ICs = {pooled_acc:.2f}",
    )
    fig.colorbar(sc, ax=ax, label="per-year lead-1 RMSE (K)")
    ax.set(
        ylim=(-1.05, 1.15),
        xlabel="Target Nino3.4 std within the year (K)",
        ylabel="Per-year lead-1 ACC (n=12, 95% CI)",
        title=(
            "(c) Per-year ACC tracks how much ENSO signal the year has,\n"
            "not the error; with n=12 the CIs span most of [-1, 1]"
        ),
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # (d) the low-ACC years month by month, on the pooled y-scale
    ax = axes[1, 1]
    worst_years = stats.nsmallest(2, "acc").job_year.astype(int).tolist()
    for color, year in zip(colors, worst_years):
        sel = data[data.job_year == year].sort_values("init_month")
        ax.plot(
            sel.init_month,
            sel.target,
            "-o",
            color=color,
            lw=1.6,
            ms=6,
            label=f"yr{year:04d} target",
        )
        ax.plot(
            sel.init_month,
            sel.prediction,
            "--s",
            color=color,
            lw=1.6,
            ms=6,
            mfc="white",
            label=f"yr{year:04d} prediction",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(
        [0.5, 12.5],
        -data.target.std(),
        data.target.std(),
        color="0.85",
        zorder=0,
        label=f"pooled target ±1 std ({data.target.std():.2f} K)",
    )
    ax.set(
        xlim=(0.5, 12.5),
        ylim=(min(lim[0], -2.6), max(lim[1], 2.6)),
        xticks=range(1, 13),
        xlabel="Initialization month",
        ylabel="Nino3.4 anomaly (K)",
        title=(
            f"(d) The two negative-ACC years, yr{worst_years[0]:04d} and "
            f"yr{worst_years[1]:04d}:\nboth stay inside the noise band all year"
        ),
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Lead-1 Nino3.4 skill, coupled FT AR SST rollouts "
        f"(model years {data.job_year.min():04d}-{data.job_year.max():04d})",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--analysis-dir", type=Path, default=here / "nino_ar_sst_analysis"
    )
    parser.add_argument("--year-start", type=int, default=231)
    parser.add_argument("--year-end", type=int, default=250)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--highlight",
        type=int,
        nargs="*",
        default=None,
        help="Years to highlight; defaults to the two lowest-ACC years.",
    )
    args = parser.parse_args()

    data = load_lead1(args.analysis_dir, args.year_start, args.year_end)
    stats = per_year_stats(data)
    highlight = tuple(
        args.highlight
        if args.highlight
        else stats.nsmallest(2, "acc").job_year.astype(int).tolist()
    )
    output = args.output or (args.analysis_dir / "lead1_diagnostics.png")
    make_figure(data, stats, output, highlight)

    pooled_acc = float(data.prediction.corr(data.target))
    pooled_rmse = float(np.sqrt(((data.prediction - data.target) ** 2).mean()))
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPooled lead-1 ACC : {pooled_acc:.3f}")
    print(f"Pooled lead-1 RMSE: {pooled_rmse:.3f} K")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
