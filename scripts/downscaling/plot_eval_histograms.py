#!/usr/bin/env python
"""
Compare evaluation histogram and power spectrum outputs from multiple beaker datasets.

Each dataset should contain an evaluator_maps_and_metrics.nc or
generated_maps_and_metrics.nc file produced by fme.downscaling.evaluator.

Usage:
    python plot_eval_histograms.py label1:BEAKER_ID1 [label2:BEAKER_ID2 ...] \
        [--output-dir ./eval_histograms] \
        [--variables VAR1 VAR2 ...] \
        [--coarse]

Requires:
    beaker CLI to be installed and authenticated (https://github.com/allenai/beaker).
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import fetch_beaker_dataset

_EVAL_FILENAMES = ["evaluator_maps_and_metrics.nc", "generated_maps_and_metrics.nc"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare evaluation histograms and power spectra across beaker runs"
    )
    parser.add_argument(
        "runs",
        nargs="+",
        metavar="LABEL:BEAKER_ID",
        help="One or more label:beaker_dataset_id pairs",
    )
    parser.add_argument(
        "--output-dir",
        default="./eval_histograms",
        help="Output directory for figures (default: ./eval_histograms)",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Filter to only these base variable names (default: all eligible)",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Show coarse power spectrum lines in power spectrum plots",
    )
    return parser.parse_args()


def find_eval_file(directory: str) -> Path | None:
    """Find an evaluator metrics NetCDF in directory or any subdirectory."""
    for fname in _EVAL_FILENAMES:
        direct = Path(directory) / fname
        if direct.exists():
            return direct
    for fname in _EVAL_FILENAMES:
        for p in sorted(Path(directory).rglob(fname)):
            return p
    return None


def detect_histogram_variables(ds: xr.Dataset) -> list[str]:
    """Return base variable names that have histogram data."""
    return sorted(
        v[len("histogram_") : -len("_bin_edges")]
        for v in map(str, ds.data_vars)
        if v.startswith("histogram_") and v.endswith("_bin_edges")
    )


def detect_power_spectrum_variables(ds: xr.Dataset) -> list[str]:
    """Return base variable names that have power spectrum data."""
    vars_: set[str] = set()
    for v in map(str, ds.data_vars):
        if "power_spectrum_fine." in v or "power_spectrum_target." in v:
            vars_.add(v.split(".")[-1])
    return sorted(vars_)


def calculate_percentile(
    counts: np.ndarray, bins: np.ndarray, percentile: float = 99.99
) -> float:
    """Interpolate a percentile value from histogram counts and bin edges."""
    total = counts.sum()
    if total == 0:
        return float(bins[0])
    cumulative = counts.cumsum() / total
    target = percentile / 100.0
    idx = int((cumulative >= target).argmax())
    if idx == 0:
        return float(bins[0])
    lower = float(bins[idx])
    upper = float(bins[idx + 1])
    width = upper - lower
    cum_lower = float(cumulative[idx - 1])
    bin_prop = float(counts[idx]) / total
    if bin_prop > 0:
        return lower + width * (target - cum_lower) / bin_prop
    return lower


def _get_histogram_arrays(
    ds: xr.Dataset, var: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return (pred_counts, pred_edges, target_counts_or_None, target_edges_or_None)."""
    counts_da = ds[f"histogram_{var}"]
    edges_da = ds[f"histogram_{var}_bin_edges"]
    if "source" in counts_da.dims:
        return (
            counts_da.sel(source="prediction").values,
            edges_da.sel(source="prediction").values,
            counts_da.sel(source="target").values,
            edges_da.sel(source="target").values,
        )
    return counts_da.values, edges_da.values, None, None


def plot_histogram_comparison(
    runs: list[tuple[str, xr.Dataset]],
    var: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    target_plotted = False
    for i, (label, ds) in enumerate(runs):
        if f"histogram_{var}" not in ds.data_vars:
            continue
        color = f"C{i}"
        pred_counts, pred_edges, target_counts, target_edges = _get_histogram_arrays(
            ds, var
        )

        # Plot the target once (from the first run that has one)
        if (
            not target_plotted
            and target_counts is not None
            and target_edges is not None
        ):
            widths = target_edges[1:] - target_edges[:-1]
            density = target_counts / widths
            p99 = calculate_percentile(target_counts, target_edges, 99.99)
            ax.step(
                target_edges[:-1],
                density,
                where="post",
                color="black",
                linestyle="-",
                linewidth=1.5,
                alpha=0.7,
                label="target",
                zorder=10,  # Ensure target is plotted above other elements
            )
            ax.axvline(
                p99,
                color="black",
                linestyle=":",
                alpha=0.6,
                linewidth=1,
                label="target 99.99th",
            )
            target_plotted = True

        widths = pred_edges[1:] - pred_edges[:-1]
        density = pred_counts / widths
        p99 = calculate_percentile(pred_counts, pred_edges, 99.99)
        ax.step(
            pred_edges[:-1],
            density,
            where="post",
            color=color,
            linestyle="-",
            linewidth=2,
            label=label,
        )
        ax.axvline(p99, color=color, linestyle=":", alpha=0.8, linewidth=1)

    ax.plot(
        [], [], linestyle=":", color="grey", linewidth=1.5, label="prediction 99.99th"
    )
    ax.set_yscale("log")
    ax.set_xlabel(var)
    ax.set_ylabel("Count / bin width")
    ax.set_title(f"{var} Distribution")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # Tight x-limits: span only bins that contain data across all runs
    x_min, x_max = np.inf, -np.inf
    for _, ds in runs:
        if f"histogram_{var}" not in ds.data_vars:
            continue
        pred_counts, pred_edges, target_counts, target_edges = _get_histogram_arrays(
            ds, var
        )
        for counts, edges in [
            (pred_counts, pred_edges),
            *([(target_counts, target_edges)] if target_counts is not None else []),
        ]:
            nz = np.where(counts > 0)[0]
            if len(nz):
                x_min = min(x_min, edges[nz[0]])
                x_max = max(x_max, edges[nz[-1] + 1])
    if np.isfinite(x_min) and np.isfinite(x_max):
        pad = 0.02 * (x_max - x_min)
        ax.set_xlim(x_min - pad, x_max + pad)

    # Transparent figure background only; axes background stays white
    fig.patch.set_alpha(0)
    fig.savefig(output_path, dpi=120, transparent=False, bbox_inches="tight")
    plt.close(fig)


def _find_power_spectrum_key(ds: xr.Dataset, kind: str, var: str) -> str | None:
    for prefix in [
        f"power_spectrum_{kind}.",
        f"single_sample_time_mean_power_spectrum_{kind}.",
    ]:
        key = f"{prefix}{var}"
        if key in ds.data_vars:
            return key
    return None


def plot_power_spectrum_comparison(
    runs: list[tuple[str, xr.Dataset]],
    var: str,
    output_path: Path,
    show_coarse: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    target_plotted = False
    for i, (label, ds) in enumerate(runs):
        if "wavenumber" not in ds.coords:
            continue
        color = f"C{i}"
        wn = ds["wavenumber"].values

        fine_key = _find_power_spectrum_key(ds, "fine", var)
        target_key = _find_power_spectrum_key(ds, "target", var)
        coarse_key = _find_power_spectrum_key(ds, "coarse", var)

        if fine_key:
            ax.loglog(
                wn,
                ds[fine_key].values,
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{label} prediction",
            )
        if not target_plotted and target_key:
            ax.loglog(
                wn,
                ds[target_key].values,
                color="black",
                linestyle="-",
                linewidth=1.5,
                alpha=0.7,
                label="target",
                zorder=10,  # Ensure target is plotted above other elements
            )
            target_plotted = True
        if show_coarse and coarse_key:
            ax.loglog(
                wn,
                ds[coarse_key].values,
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{label} coarse",
            )

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(f"{var} Power Spectrum")
    ax.legend()
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.patch.set_alpha(0)
    fig.savefig(output_path, dpi=120, transparent=False, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        loaded_runs: list[tuple[str, xr.Dataset]] = []

        for run_spec in args.runs:
            if ":" not in run_spec:
                raise ValueError(f"Run spec must be label:beaker_id, got: {run_spec!r}")
            label, beaker_id = run_spec.split(":", 1)
            run_dir = Path(temp_dir) / label
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"Fetching {label}: {beaker_id}")
            fetch_beaker_dataset(beaker_id, str(run_dir))

            nc_file = find_eval_file(str(run_dir))
            if nc_file is None:
                print(f"  Warning: no eval metrics file found for {label}, skipping")
                continue
            print(f"  Loading {nc_file.name}")
            loaded_runs.append((label, xr.open_dataset(nc_file)))

        if not loaded_runs:
            print("No valid runs loaded. Exiting.")
            return

        hist_vars: set[str] = set()
        ps_vars: set[str] = set()
        for _, ds in loaded_runs:
            hist_vars |= set(detect_histogram_variables(ds))
            ps_vars |= set(detect_power_spectrum_variables(ds))

        if args.variables:
            hist_vars &= set(args.variables)
            ps_vars &= set(args.variables)

        for var in sorted(hist_vars):
            out_path = output_dir / f"{var}_histogram.png"
            print(f"Plotting histogram: {var} -> {out_path}")
            plot_histogram_comparison(loaded_runs, var, out_path)

        for var in sorted(ps_vars):
            out_path = output_dir / f"{var}_power_spectrum.png"
            print(f"Plotting power spectrum: {var} -> {out_path}")
            plot_power_spectrum_comparison(
                loaded_runs, var, out_path, show_coarse=args.coarse
            )

        for _, ds in loaded_runs:
            ds.close()

    print(f"Done! Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
