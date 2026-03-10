#!/usr/bin/env python
"""
Compare histogram distributions across multiple beaker datasets.

For each event that exists in all provided datasets, generates a histogram
figure per variable showing the target distribution and the average predicted
distribution from each dataset in a different color.

Reuses data-fetching and parsing utilities from plot_beaker_histograms.py.

Usage:
    python compare_histograms.py <dataset_id_1> <dataset_id_2> [<dataset_id_3> ...]
        [--labels label1 label2 ...] [--output-dir <path>]

Requires:
    beaker CLI to be installed and authenticated.
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from plot_beaker_histograms import (
    detect_variable_pairs,
    fetch_beaker_dataset,
    find_event_files,
)

COLORS = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare histogram distributions across beaker datasets"
    )
    parser.add_argument(
        "beaker_dataset_ids",
        nargs="+",
        help="Two or more beaker dataset IDs to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each dataset (defaults to dataset IDs)",
    )
    parser.add_argument(
        "--output-dir",
        default="./histogram_comparison",
        help="Output directory for figures (default: ./histogram_comparison)",
    )
    args = parser.parse_args()
    if len(args.beaker_dataset_ids) < 2:
        parser.error("At least 2 beaker_dataset_ids are required")
    if args.labels is not None and len(args.labels) != len(args.beaker_dataset_ids):
        parser.error("Number of --labels must match number of dataset IDs")
    return args


def plot_comparison_histogram(
    datasets: list[xr.Dataset],
    labels: list[str],
    var_prefix: str,
    event_name: str,
    save_path: Path,
    n_bins: int = 50,
) -> None:
    """
    Plot overlaid histograms comparing the average predicted distribution
    from each dataset against the target.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    target_data = datasets[0][f"{var_prefix}_target"]
    predicted_per_dataset = [ds[f"{var_prefix}_predicted"].values for ds in datasets]

    all_values = np.concatenate(
        [target_data.values.ravel()] + [p.ravel() for p in predicted_per_dataset]
    )
    data_min, data_max = np.nanmin(all_values), np.nanmax(all_values)
    bin_edges = np.linspace(data_min, data_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    target_counts, _ = np.histogram(target_data.values.ravel(), bins=bin_edges)
    ax.step(
        bin_centers,
        target_counts,
        where="mid",
        color="black",
        linewidth=2,
        label="Target",
    )

    for i, (pred_data, label) in enumerate(zip(predicted_per_dataset, labels)):
        if pred_data.ndim != 3:
            raise ValueError(
                f"Expected predicted data to be 3D (samples, lat, lon), "
                f"got shape {pred_data.shape} for dataset '{label}'"
            )
        sample_counts = []
        for s in range(pred_data.shape[0]):
            counts, _ = np.histogram(pred_data[s].ravel(), bins=bin_edges)
            sample_counts.append(counts)
        avg_counts = np.mean(sample_counts, axis=0)

        color = COLORS[i % len(COLORS)]
        ax.step(
            bin_centers,
            avg_counts,
            where="mid",
            color=color,
            linewidth=2,
            label=f"{label} (avg predicted)",
        )

    var_label = var_prefix.replace("_", " ").title()
    ax.set_xlabel(var_label)
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ylim_max = 10 ** (np.log10(max(np.max(target_counts), 1)) + 1)
    ax.set_ylim(0.1, ylim_max)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"{event_name} — {var_label}")
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    variables = detect_variable_pairs(ds)
    if (
        "eastward_wind_at_ten_meters" in variables
        and "northward_wind_at_ten_meters" in variables
    ):
        ds["wind_speed_target"] = np.sqrt(
            ds.eastward_wind_at_ten_meters_target**2
            + ds.northward_wind_at_ten_meters_target**2
        )
        ds["wind_speed_predicted"] = np.sqrt(
            ds.eastward_wind_at_ten_meters_predicted**2
            + ds.northward_wind_at_ten_meters_predicted**2
        )
    return ds


def main():
    args = parse_args()
    dataset_ids = args.beaker_dataset_ids
    labels = args.labels or dataset_ids
    output_dir = Path(args.output_dir)

    event_files_per_dataset: list[dict[str, xr.Dataset]] = []

    temp_dirs = []
    try:
        for ds_id in dataset_ids:
            td = tempfile.mkdtemp()
            temp_dirs.append(td)
            print(f"Fetching beaker dataset: {ds_id}")
            fetch_beaker_dataset(ds_id, td)
            event_files = find_event_files(td)
            print(f"  Found {len(event_files)} event(s): {list(event_files.keys())}")
            event_files_per_dataset.append(event_files)

        common_events = set(event_files_per_dataset[0].keys())
        for ef in event_files_per_dataset[1:]:
            common_events &= set(ef.keys())
        sorted_events = sorted(common_events)

        if not sorted_events:
            print("No common events found across all datasets.")
            return

        print(f"\nCommon events ({len(sorted_events)}): {sorted_events}")

        for event_name in sorted_events:
            opened_datasets = []
            for ef in event_files_per_dataset:
                event_ds = xr.open_dataset(ef[event_name])
                event_ds = add_wind_speed(event_ds)
                opened_datasets.append(event_ds)

            all_var_sets = [set(detect_variable_pairs(ds)) for ds in opened_datasets]
            common_vars = sorted(set.intersection(*all_var_sets))

            if not common_vars:
                print(f"  No common variable pairs for event {event_name}, skipping.")
                for ds in opened_datasets:
                    ds.close()
                continue

            for var_prefix in common_vars:
                fig_path = output_dir / event_name / f"{var_prefix}.png"
                plot_comparison_histogram(
                    opened_datasets,
                    labels,
                    var_prefix,
                    event_name,
                    save_path=fig_path,
                )
                print(f"  Saved: {fig_path}")

            for ds in opened_datasets:
                ds.close()
    finally:
        import shutil

        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)

    print("Done!")


if __name__ == "__main__":
    main()
