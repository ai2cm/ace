"""
Plot a histogram of per-track maximum wind speed from detect_tc_tracks output.

Reads one or more ``tracks.csv`` files written by ``detect_tc_tracks.py`` (each
with a ``track_id`` and ``wind`` column, one row per track point), reduces each
track to its single maximum wind speed, and plots the distribution of those
per-track maxima as a histogram. Given multiple input files, all histograms are
overlaid on the same figure so distributions can be compared directly.

Usage examples:
    # Single run
    python plot_max_wind_histogram.py out/tracks.csv --out max_wind_hist.png

    # Compare two runs on the same axes (auto-labelled by parent dir name)
    python plot_max_wind_histogram.py runA/tracks.csv runB/tracks.csv \
        --out max_wind_hist.png

    # Explicit legend labels
    python plot_max_wind_histogram.py runA/tracks.csv runB/tracks.csv \
        --labels ACE X-SHiELD --out max_wind_hist.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def per_track_max_wind(csv_path: Path, wind_col: str = "wind") -> pd.Series:
    """Return the maximum wind speed for each track in a tracks.csv file.

    Groups the (one-row-per-track-point) tracks table by ``track_id`` and takes
    the max of ``wind_col``, yielding one value per track.
    """
    df = pd.read_csv(csv_path)
    for col in ("track_id", wind_col):
        if col not in df.columns:
            raise KeyError(
                f"{col!r} column not found in {csv_path}; available: "
                f"{list(df.columns)}"
            )
    max_wind = df.groupby("track_id")[wind_col].max()
    logger.info(
        "%s: %d tracks, max wind range %.1f-%.1f m/s",
        csv_path,
        len(max_wind),
        max_wind.min() if len(max_wind) else float("nan"),
        max_wind.max() if len(max_wind) else float("nan"),
    )
    return max_wind


def _default_label(csv_path: Path) -> str:
    """Label a series by its parent directory name, or file stem at top level."""
    parent = csv_path.parent.name
    return parent if parent else csv_path.stem


def plot_histograms(
    max_winds: list[pd.Series],
    labels: list[str],
    bins: int,
    out_path: Path,
    wind_col: str = "wind",
) -> None:
    """Overlay per-track max-wind histograms for each input on one figure."""
    # Share a common bin edge set so overlaid histograms are directly
    # comparable rather than each series binning over its own range.
    all_vals = np.concatenate([mw.values for mw in max_winds if len(mw)])
    if len(all_vals) == 0:
        raise ValueError("No tracks found in any input file; nothing to plot.")
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for max_wind, label in zip(max_winds, labels):
        ax.hist(
            max_wind.values,
            bins=bin_edges,
            alpha=0.5 if len(max_winds) > 1 else 1.0,
            label=f"{label} (n={len(max_wind)})",
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xlabel("Maximum wind speed per track (m/s)")
    ax.set_ylabel("Number of tracks")
    ax.set_title("Distribution of tropical cyclone peak wind speed")
    if len(max_winds) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Wrote histogram to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tracks_csv",
        nargs="+",
        help="One or more tracks.csv files from detect_tc_tracks.py.",
    )
    parser.add_argument(
        "--out",
        default="max_wind_histogram.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels, one per input file (default: parent dir names).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--wind-col",
        default="wind",
        help="Name of the wind-speed column in the tracks CSV.",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.tracks_csv]
    if args.labels is not None and len(args.labels) != len(csv_paths):
        parser.error(
            f"--labels got {len(args.labels)} label(s) for {len(csv_paths)} "
            "input file(s); counts must match."
        )
    labels = args.labels or [_default_label(p) for p in csv_paths]

    max_winds = [per_track_max_wind(p, wind_col=args.wind_col) for p in csv_paths]
    plot_histograms(
        max_winds,
        labels,
        bins=args.bins,
        out_path=Path(args.out),
        wind_col=args.wind_col,
    )


if __name__ == "__main__":
    main()
