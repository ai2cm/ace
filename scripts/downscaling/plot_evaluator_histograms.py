#!/usr/bin/env python
"""
Fetch evaluator_maps_and_metrics.nc from multiple beaker datasets and plot
their histogram distributions overlaid for comparison.

The evaluator output file contains histogram data (counts and bin edges) for
each variable, stored under "histogram_<var_name>" with a "source" dimension
that distinguishes "target" from "prediction" histograms.

Usage:
    python plot_evaluator_histograms.py <dataset_id_1> <dataset_id_2> [...]
        [--labels label1 label2 ...] [--output-dir <path>]

Requires:
    beaker CLI to be installed and authenticated.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import fetch_beaker_dataset

from fme.core.histogram import _normalize_histogram, trim_zero_bins

NC_FILENAME = "evaluator_maps_and_metrics.nc"

COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

_EASTWARD_WIND = "eastward_wind_at_ten_meters"
_NORTHWARD_WIND = "northward_wind_at_ten_meters"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch evaluator_maps_and_metrics.nc from multiple beaker datasets "
            "and plot overlaid histograms."
        )
    )
    parser.add_argument(
        "beaker_dataset_ids",
        nargs="+",
        help="Beaker dataset IDs to fetch and compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each dataset (defaults to dataset IDs)",
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluator_histograms",
        help="Output directory for figures (default: ./evaluator_histograms)",
    )
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.beaker_dataset_ids):
        parser.error("Number of --labels must match number of dataset IDs")
    return args


def add_wind_speed_histogram(ds: xr.Dataset, n_bins: int = 300) -> xr.Dataset:
    """Add a wind_speed histogram derived from time-mean wind component maps.

    Computes sqrt(u^2 + v^2) point-wise from the time-mean maps of eastward
    and northward wind, then histograms the result for both target and
    prediction sources.  Returns the dataset unchanged if the required map
    variables are missing.
    """
    map_keys = {
        "target": (
            f"time_mean_target.{_EASTWARD_WIND}",
            f"time_mean_target.{_NORTHWARD_WIND}",
        ),
        "prediction": (
            f"time_mean_prediction.{_EASTWARD_WIND}",
            f"time_mean_prediction.{_NORTHWARD_WIND}",
        ),
    }
    if not all(k in ds for pair in map_keys.values() for k in pair):
        return ds

    counts_list, edges_list = [], []
    for source in ["target", "prediction"]:
        u_key, v_key = map_keys[source]
        ws = np.sqrt(ds[u_key].values ** 2 + ds[v_key].values ** 2).ravel()
        ws = ws[~np.isnan(ws)]
        counts, bin_edges = np.histogram(ws, bins=n_bins)
        counts_list.append(counts)
        edges_list.append(bin_edges)

    ds["histogram_wind_speed"] = xr.DataArray(
        np.stack(counts_list),
        dims=("source", "bin"),
    )
    ds["histogram_wind_speed_bin_edges"] = xr.DataArray(
        np.stack(edges_list),
        dims=("source", "bin_edges"),
    )
    return ds


def detect_histogram_variables(ds: xr.Dataset) -> list[str]:
    """Find histogram variable names that have matching _bin_edges variables."""
    hist_vars = []
    for var in ds.data_vars:
        if var.startswith("histogram_") and not var.endswith("_bin_edges"):
            if f"{var}_bin_edges" in ds:
                hist_vars.append(var)
    return sorted(hist_vars)


def _plot_single_histogram(
    ax: plt.Axes,
    ds: xr.Dataset,
    var_name: str,
    source: str,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.5,
) -> None:
    counts = ds[var_name].sel(source=source).values
    bin_edges = ds[f"{var_name}_bin_edges"].sel(source=source).values
    counts, bin_edges = trim_zero_bins(counts, bin_edges)
    normalized = _normalize_histogram(counts, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mask = normalized > 0
    ax.plot(
        bin_centers[mask],
        normalized[mask],
        linestyle,
        label=label,
        color=color,
        linewidth=linewidth,
    )


def plot_overlaid_histograms(
    datasets: list[xr.Dataset],
    labels: list[str],
    var_name: str,
    save_path: Path,
) -> None:
    """Plot target and prediction histograms from multiple datasets on one axis.

    The target histogram is taken from the first dataset that contains the
    variable and drawn as a solid black line. Each dataset's prediction
    histogram is drawn in a distinct color. Datasets that do not contain the
    variable are silently skipped.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    target_plotted = False
    color_idx = 0
    for ds, label in zip(datasets, labels):
        hist_vars = detect_histogram_variables(ds)
        if var_name not in hist_vars:
            continue
        if not target_plotted:
            _plot_single_histogram(
                ax,
                ds,
                var_name,
                "target",
                label="Target",
                color="black",
                linewidth=2,
            )
            target_plotted = True
        color = COLORS[color_idx % len(COLORS)]
        _plot_single_histogram(
            ax,
            ds,
            var_name,
            "prediction",
            label=label,
            color=color,
        )
        color_idx += 1

    display_name = var_name.removeprefix("histogram_").replace("_", " ").title()
    ax.set_xlabel(display_name)
    ax.set_ylabel("Probability Density")
    ax.set_yscale("log")
    ax.set_title(display_name)
    ax.legend()
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    dataset_ids = args.beaker_dataset_ids
    labels = args.labels or dataset_ids
    output_dir = Path(args.output_dir)

    datasets: list[xr.Dataset] = []
    temp_dirs: list[str] = []

    try:
        for ds_id in dataset_ids:
            td = tempfile.mkdtemp()
            temp_dirs.append(td)
            print(f"Fetching beaker dataset: {ds_id}")
            result_dir = fetch_beaker_dataset(ds_id, td, prefix=NC_FILENAME)
            nc_path = Path(result_dir) / NC_FILENAME
            if not nc_path.exists():
                raise FileNotFoundError(
                    f"{NC_FILENAME} not found in dataset {ds_id} at {nc_path}"
                )
            ds = xr.open_dataset(nc_path)
            ds = add_wind_speed_histogram(ds)
            datasets.append(ds)

        all_vars: set[str] = set()
        for ds in datasets:
            all_vars.update(detect_histogram_variables(ds))
        all_vars_sorted = sorted(all_vars)

        if not all_vars_sorted:
            print("No histogram variables found in any dataset.")
            return

        print(f"Found {len(all_vars_sorted)} histogram variable(s):")
        for v in all_vars_sorted:
            present_in = [
                lbl
                for lbl, ds in zip(labels, datasets)
                if v in detect_histogram_variables(ds)
            ]
            print(f"  {v}  (in {len(present_in)}/{len(datasets)} datasets)")

        for var_name in all_vars_sorted:
            fig_path = output_dir / f"{var_name}.png"
            plot_overlaid_histograms(datasets, labels, var_name, save_path=fig_path)
            print(f"  Saved: {fig_path}")

    finally:
        for ds in datasets:
            ds.close()
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)

    print("Done!")


if __name__ == "__main__":
    main()
