#!/usr/bin/env python
"""
Compare zonal power spectra across multiple downscaling evaluation runs.

Each ``fme.downscaling.evaluator`` run writes an
``evaluator_maps_and_metrics.nc`` to its beaker result dataset. That file holds
the mean zonal power spectrum (computed by
``fme.downscaling.aggregators.main.ZonalPowerSpectrumComparison``) for the
prediction, the target, and the interpolated coarse input, keyed per variable:

    power_spectrum_fine.<VAR>     # model prediction (fine grid)
    power_spectrum_target.<VAR>   # ground-truth target (fine grid)
    power_spectrum_coarse.<VAR>   # interpolated coarse input (context)

on a shared ``wavenumber`` axis. This script fetches that file for each run,
overlays the prediction spectra against the (shared) target, and — optionally —
plots the ratio to target so spectral fidelity is read as a deviation from 1.

The defaults compare the original Hiro v1 diffusion model, the GAN-only
distilled baseline, and the spectral-loss distilled student on the CONUS
X-SHiELD AMIP-control evaluation (100km -> 3km, best_student_tail.ckpt).

Usage:
    # defaults (hiro v1 / baseline / spectral, CONUS PRATEsfc):
    python plot_compared_power_spectra.py

    # explicit datasets + labels:
    python plot_compared_power_spectra.py \
        --datasets <id1> <id2> [...] --labels hiro baseline spectral \
        --variables PRATEsfc --output-dir ./psd_comparison

Requires:
    beaker CLI installed and authenticated (https://github.com/allenai/beaker).
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import fetch_beaker_dataset

METRICS_FILENAME = "evaluator_maps_and_metrics.nc"

# Prefixes used by ZonalPowerSpectrumComparison, after the evaluator rewrites
# "/" -> "_" for netCDF variable names.
FINE_PREFIX = "power_spectrum_fine."
TARGET_PREFIX = "power_spectrum_target."
COARSE_PREFIX = "power_spectrum_coarse."

# CONUS X-SHiELD AMIP-control eval (100km -> 3km, best_student_tail.ckpt).
# Beaker result-dataset IDs, resolved from each eval run's experiment.
DEFAULT_DATASETS = [
    "01KT2JXK50MKQSAEX3YY75WSTP",  # hiro v1 diffusion (wandb j3thqivd)
    "01KX6TBGQY8E99J36M008QSJ2B",  # GAN-only baseline (eval flzvb6tp; f7z93y0a)
    "01KX6TBNJRQD25HEA98QH0KP08",  # spectral distilled (eval x2nyzmzh; i26sidsm)
]
DEFAULT_LABELS = [
    "Hiro v1 (diffusion)",
    "Distilled baseline (GAN-only)",
    "Distilled spectral",
]

COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare zonal power spectra across evaluation beaker datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Beaker result-dataset IDs to compare "
        "(default: hiro v1 / baseline / spectral CONUS)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each dataset "
        "(defaults to the built-in names, else dataset IDs)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variable name(s) to plot, e.g. PRATEsfc (default: all common variables)",
    )
    parser.add_argument(
        "--output-dir",
        default="./power_spectrum_comparison",
        help="Output directory for figures (default: ./power_spectrum_comparison)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache fetched datasets (or set BEAKER_DATASET_CACHE_DIR)",
    )
    parser.add_argument(
        "--no-coarse",
        action="store_true",
        help="Do not plot the interpolated-coarse input spectrum",
    )
    parser.add_argument(
        "--no-ratio",
        action="store_true",
        help="Only plot the spectra, without the prediction/target ratio panel",
    )
    args = parser.parse_args()
    if args.labels is None:
        if args.datasets == DEFAULT_DATASETS:
            args.labels = DEFAULT_LABELS
        else:
            args.labels = args.datasets
    if len(args.labels) != len(args.datasets):
        parser.error("Number of --labels must match number of --datasets")
    return args


def load_metrics_dataset(dataset_id: str, cache_dir: str | None) -> xr.Dataset:
    """Fetch just the metrics netCDF from a beaker dataset and open it."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = fetch_beaker_dataset(
            dataset_id, temp_dir, prefix=METRICS_FILENAME, cache_dir=cache_dir
        )
        nc_path = Path(data_dir) / METRICS_FILENAME
        if not nc_path.is_file():
            raise FileNotFoundError(
                f"{METRICS_FILENAME} not found in dataset {dataset_id} "
                f"(searched {data_dir})"
            )
        # load() so the dataset survives the temp dir cleanup on cache miss.
        return xr.open_dataset(nc_path).load()


def spectrum_variables(ds: xr.Dataset) -> list[str]:
    """Variable names that have a prediction (fine) power spectrum."""
    return sorted(
        v[len(FINE_PREFIX) :] for v in ds.data_vars if v.startswith(FINE_PREFIX)
    )


def wavenumbers(ds: xr.Dataset, n: int) -> np.ndarray:
    if "wavenumber" in ds.coords:
        return np.asarray(ds["wavenumber"].values)
    return np.arange(n)


def plot_variable(
    datasets: list[xr.Dataset],
    labels: list[str],
    var: str,
    save_path: Path,
    show_coarse: bool = True,
    show_ratio: bool = True,
) -> None:
    """Overlay each run's prediction spectrum against the shared target."""
    # Reference target/coarse taken from the first dataset (all runs share the
    # same evaluation data, so the target spectra are identical).
    ref = datasets[0]
    target = np.asarray(ref[f"{TARGET_PREFIX}{var}"].values)
    k = wavenumbers(ref, len(target))
    # Skip k=0 (domain mean / DC term) so the log-log axes are well defined.
    sl = slice(1, None)

    if show_ratio:
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(9, 8), sharex=True, height_ratios=[3, 1]
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax_r = None

    ax.loglog(
        k[sl], target[sl], color="black", linewidth=2.0, label="Target (X-SHiELD 3km)"
    )
    if show_coarse and f"{COARSE_PREFIX}{var}" in ref.data_vars:
        coarse = np.asarray(ref[f"{COARSE_PREFIX}{var}"].values)
        ax.loglog(
            k[sl],
            coarse[sl],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Coarse input (interpolated)",
        )

    for i, (ds, label) in enumerate(zip(datasets, labels)):
        pred = np.asarray(ds[f"{FINE_PREFIX}{var}"].values)
        color = COLORS[i % len(COLORS)]
        ax.loglog(
            k[sl], pred[sl], color=color, linestyle="-", linewidth=1.8, label=label
        )
        if ax_r is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = pred[sl] / target[sl]
            ax_r.semilogx(k[sl], ratio, color=color, linewidth=1.5, label=label)

    ax.set_ylabel("Power spectral density")
    ax.set_title(f"Power spectrum: {var}")
    ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.legend()

    if ax_r is not None:
        ax_r.axhline(1.0, color="black", linewidth=1.0)
        ax_r.set_ylabel("prediction / target")
        ax_r.set_xlabel("Wavenumber")
        ax_r.set_ylim(0, 2)
        ax_r.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.3)
    else:
        ax.set_xlabel("Wavenumber")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    datasets: list[xr.Dataset] = []
    for ds_id, label in zip(args.datasets, args.labels):
        print(f"Fetching {label} ({ds_id})")
        datasets.append(load_metrics_dataset(ds_id, args.cache_dir))

    var_sets = [set(spectrum_variables(ds)) for ds in datasets]
    common_vars = sorted(set.intersection(*var_sets))
    if args.variables is not None:
        missing = [v for v in args.variables if v not in common_vars]
        if missing:
            raise SystemExit(
                f"Requested variables not present in all datasets: {missing}. "
                f"Available: {common_vars}"
            )
        common_vars = list(args.variables)
    if not common_vars:
        raise SystemExit("No power-spectrum variables common to all datasets.")

    print(f"Plotting variables: {common_vars}")
    for var in common_vars:
        save_path = output_dir / f"power_spectrum_{var}.png"
        plot_variable(
            datasets,
            args.labels,
            var,
            save_path=save_path,
            show_coarse=not args.no_coarse,
            show_ratio=not args.no_ratio,
        )
        print(f"  Saved: {save_path}")

    for ds in datasets:
        ds.close()
    print("Done!")


if __name__ == "__main__":
    main()
