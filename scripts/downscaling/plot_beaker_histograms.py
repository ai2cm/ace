#!/usr/bin/env python
"""
Fetch netCDF event files from a beaker dataset and generate histogram plots
comparing ensemble predictions against targets for each variable.

This will work for saved event outputs from `fme.downscaling.evaluator`
from a beaker experiment.  It downloads the experiment files to a temporary
directory and then parses the filenames for <event_name>_YYYYMMDD.nc to look
for single-event outputs.

Usage:
    python plot_beaker_histograms.py <beaker_dataset_id> [--output-dir <path>]

Requires:
    beaker CLI to be installed and authenticated (https://github.com/allenai/beaker).
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Matching for <event_name>_YYYYMMDD.nc
_EVENT_FILE_RE = re.compile(r"(.+)_(\d{8})\.nc$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate histogram plots from beaker dataset event files"
    )
    parser.add_argument(
        "beaker_dataset_id",
        help="The beaker dataset ID to fetch",
    )
    parser.add_argument(
        "--output-dir",
        default="./histogram_outputs",
        help="Output directory for figures (default: ./histogram_outputs)",
    )
    return parser.parse_args()


def fetch_beaker_dataset(dataset_id: str, target_dir: str) -> None:
    """Fetch a beaker dataset to the specified directory."""
    subprocess.run(
        ["beaker", "dataset", "fetch", dataset_id, "--output", target_dir],
        check=True,
    )


def find_event_files(directory: str) -> dict[str, Path]:
    """Find netCDF files matching the event naming pattern, keyed by event name."""
    event_files = {}
    for p in sorted(Path(directory).glob("*.nc")):
        # extract event name
        matched = _EVENT_FILE_RE.match(p.name)
        if matched:
            event_files[matched.group(1)] = p
    return event_files


def detect_variable_pairs(ds: xr.Dataset) -> list[str]:
    """Detect variables that have both _predicted and _target versions."""
    predicted = {
        v[: -len("_predicted")] for v in ds.data_vars if v.endswith("_predicted")
    }
    target = {v[: -len("_target")] for v in ds.data_vars if v.endswith("_target")}
    return sorted(predicted & target)


def plot_histogram_lines(
    ds: xr.Dataset,
    key_prefix: str,
    title_prefix: str,
    save_path: Path,
) -> None:
    """
    Plot histogram comparing ensemble predictions against target.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gather all predicted and target data
    target_data = ds[f"{key_prefix}_target"]
    predicted_data = ds[f"{key_prefix}_predicted"]
    all_data = np.concatenate([target_data.values[None], predicted_data.values])

    # Compute bin edges from min/max of all data
    bins = 50
    data_min, data_max = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    sample_data = predicted_data.values
    if sample_data.ndim != 3:
        raise ValueError(
            f"Expected predicted data to be 3D (samples, lat, lon), "
            f"got shape {sample_data.shape}"
        )

    # Calculate the tail percentile values for each generated sample
    # and generate a 95% confidence interval
    lower_bounds = np.percentile(sample_data, 0.01, axis=(1, 2))
    upper_bounds = np.percentile(sample_data, 99.99, axis=(1, 2))
    lower_bound_2p5 = np.percentile(lower_bounds, 2.5)
    lower_bound_97p5 = np.percentile(lower_bounds, 97.5)
    upper_bound_2p5 = np.percentile(upper_bounds, 2.5)
    upper_bound_97p5 = np.percentile(upper_bounds, 97.5)

    # Calculate target percentiles
    target_lower_0p01_percentile = np.percentile(target_data.values, 0.01)
    target_upper_99p99_percentile = np.percentile(target_data.values, 99.99)
    counts, _ = np.histogram(target_data.values, bins=bin_edges)

    # Pre-compute y-axis limits so fill_betweenx spans exactly the plot area
    ylim_min = 0.1
    ylim_max = 10 ** (np.log10(np.max(counts)) + 1)

    # Calculate histogram for each predicted sample
    all_counts = []
    num_samples = sample_data.shape[0]
    for i in range(num_samples):
        sample_flat = sample_data[i].flatten()
        sample_counts, _ = np.histogram(sample_flat, bins=bin_edges)
        ax.step(
            bin_centers,
            sample_counts,
            where="mid",
            alpha=0.1,
            label="Samples" if i == 0 else None,
        )
        all_counts.append(sample_counts)
    all_counts = np.stack(all_counts)

    ax.step(
        bin_centers, counts, where="mid", color="black", linewidth=2, label="Target"
    )
    ax.axvline(
        target_lower_0p01_percentile,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label="Target 0.01%",
    )
    ax.axvline(
        target_upper_99p99_percentile,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label="Target 99.99%",
    )
    ax.fill_betweenx(
        [ylim_min, ylim_max],
        upper_bound_2p5,
        upper_bound_97p5,
        color="gray",
        alpha=0.2,
        label="Pred upper tail 95% CI",
    )
    ax.fill_betweenx(
        [ylim_min, ylim_max],
        lower_bound_2p5,
        lower_bound_97p5,
        color="gray",
        alpha=0.2,
        label="Pred lower tail 95% CI",
    )

    avg_counts = np.mean(all_counts, axis=0)
    ax.step(
        bin_centers,
        avg_counts,
        where="mid",
        color="C0",
        linewidth=2,
        label="Average Predicted",
    )
    var_label = key_prefix.replace("_", " ").title()
    ax.set_xlabel(var_label)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"{title_prefix} distribution: {var_label}")
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    beaker_id = args.beaker_dataset_id
    output_dir = Path(args.output_dir)

    print(f"Fetching beaker dataset: {beaker_id}")

    with tempfile.TemporaryDirectory() as temp_dir:
        fetch_beaker_dataset(beaker_id, temp_dir)

        event_files = find_event_files(temp_dir)
        if not event_files:
            print(f"No event files found in dataset {beaker_id}")
            return

        print(f"Found {len(event_files)} event file(s)")

        for event_name, nc_file in event_files.items():
            output_event_dir = output_dir / beaker_id / event_name

            print(f"Processing: {nc_file.name} -> {output_event_dir}")

            ds = xr.open_dataset(nc_file)
            variables = detect_variable_pairs(ds)

            if not variables:
                print(f"  No variable pairs found in {nc_file.name}")
                continue

            for var_prefix in variables:
                fig_path = output_event_dir / f"{var_prefix}.png"
                plot_histogram_lines(ds, var_prefix, event_name, save_path=fig_path)
                print(f"  Saved: {fig_path}")

            ds.close()

    print("Done!")


if __name__ == "__main__":
    main()
