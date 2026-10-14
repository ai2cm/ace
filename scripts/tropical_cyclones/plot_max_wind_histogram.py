"""
Plot per-track peak-intensity histograms from detect_tc_tracks output.

Reads one or more ``tracks.csv`` files written by ``detect_tc_tracks.py`` (each
with ``track_id``, ``wind``, and ``slp`` columns, one row per track point),
reduces each track to two peak-intensity values, and plots their distributions
side by side:

  * Maximum wind speed per track.
  * Minimum sea-level pressure per track.

Given multiple input files, all histograms are overlaid on the same axes so
distributions can be compared directly.

Usage examples:
    # Single run
    python plot_max_wind_histogram.py out/tracks.csv --out peak_intensity_hist.png

    # Compare two runs on the same axes (auto-labelled by parent dir name)
    python plot_max_wind_histogram.py runA/tracks.csv runB/tracks.csv \
        --out peak_intensity_hist.png

    # Explicit legend labels
    python plot_max_wind_histogram.py runA/tracks.csv runB/tracks.csv \
        --labels ACE X-SHiELD --out peak_intensity_hist.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def per_track_peaks(
    csv_path: Path,
    wind_col: str = "wind",
    slp_col: str = "slp",
) -> pd.DataFrame:
    """Reduce each track to its peak wind and minimum sea-level pressure.

    Groups the (one-row-per-track-point) tracks table by ``track_id`` and takes
    the maximum of ``wind_col`` and the minimum of ``slp_col``, yielding one
    value of each per track.

    Returns a DataFrame indexed by ``track_id`` with columns ``max_wind`` (m/s)
    and ``min_slp`` (same units as the input, typically Pa).
    """
    df = pd.read_csv(csv_path)
    for col in ("track_id", wind_col, slp_col):
        if col not in df.columns:
            raise KeyError(
                f"{col!r} column not found in {csv_path}; available: "
                f"{list(df.columns)}"
            )

    grouped = df.groupby("track_id")
    peaks = pd.DataFrame(
        {
            "max_wind": grouped[wind_col].max(),
            "min_slp": grouped[slp_col].min(),
        }
    )
    if len(peaks):
        logger.info(
            "%s: %d tracks, max wind %.1f-%.1f m/s, min slp %.0f-%.0f (input units)",
            csv_path,
            len(peaks),
            peaks["max_wind"].min(),
            peaks["max_wind"].max(),
            peaks["min_slp"].min(),
            peaks["min_slp"].max(),
        )
    else:
        logger.info("%s: 0 tracks", csv_path)
    return peaks


def _default_label(csv_path: Path) -> str:
    """Label a series by its parent directory name, or file stem at top level."""
    parent = csv_path.parent.name
    return parent if parent else csv_path.stem


def _overlay_hist(
    ax: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    bins: int,
    xlabel: str,
    show_legend: bool,
) -> None:
    """Overlay step histograms for each input series on one axis.

    Uses a common bin-edge set across inputs so the overlaid histograms are
    directly comparable rather than each series binning over its own range.
    """
    all_vals = np.concatenate([v for v in values if len(v)])
    if len(all_vals) == 0:
        raise ValueError("No tracks found in any input file; nothing to plot.")
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    for vals, label in zip(values, labels):
        ax.hist(
            vals,
            bins=bin_edges,
            histtype="step",
            linewidth=1.5,
            label=f"{label} (n={len(vals)})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of tracks")
    if show_legend:
        ax.legend()


def plot_histograms(
    peaks: list[pd.DataFrame],
    labels: list[str],
    bins: int,
    out_path: Path,
) -> None:
    """Plot per-track peak-wind and min-SLP histograms side by side."""
    multiple = len(peaks) > 1
    fig, (ax_wind, ax_slp) = plt.subplots(1, 2, figsize=(13, 5))

    _overlay_hist(
        ax_wind,
        [p["max_wind"].values for p in peaks],
        labels,
        bins,
        "Maximum wind speed per track (m/s)",
        show_legend=multiple,
    )
    ax_wind.set_title("Peak wind speed")

    # SLP is stored in Pa; plot in hPa, the meteorological convention.
    _overlay_hist(
        ax_slp,
        [p["min_slp"].values / 100.0 for p in peaks],
        labels,
        bins,
        "Minimum sea-level pressure per track (hPa)",
        show_legend=multiple,
    )
    ax_slp.set_title("Minimum sea-level pressure")

    fig.suptitle("Distribution of tropical cyclone peak intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Wrote histograms to %s", out_path)


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
        default="peak_intensity_histogram.png",
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
    parser.add_argument(
        "--slp-col",
        default="slp",
        help="Name of the sea-level-pressure column in the tracks CSV.",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.tracks_csv]
    if args.labels is not None and len(args.labels) != len(csv_paths):
        parser.error(
            f"--labels got {len(args.labels)} label(s) for {len(csv_paths)} "
            "input file(s); counts must match."
        )
    labels = args.labels or [_default_label(p) for p in csv_paths]

    peaks = [
        per_track_peaks(p, wind_col=args.wind_col, slp_col=args.slp_col)
        for p in csv_paths
    ]
    plot_histograms(
        peaks,
        labels,
        bins=args.bins,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
