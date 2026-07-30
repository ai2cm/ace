"""Diagnose an eval run's precipitation histogram and zonal power spectrum.

Reads ``evaluator_maps_and_metrics.nc`` (written by ``fme.downscaling.evaluator``)
and reports, per magnitude band and per wavenumber, how the prediction compares
to the target.  Motivated by two f-distill defects that the wandb scalars hide:
an over-produced 200-400 mm/day precipitation range, and a power-spectrum excess
at low wavenumber.  See ``fme/downscaling/distillation/experiments/LOG.md``.

**The bin-edge trap.** ``ComparedDynamicHistograms`` gives the target and the
prediction *independent* ``DynamicHistogramAggregator``s, so their bin edges
differ (a wider-tailed prediction gets wider bins).  Comparing raw per-bin counts
across sources, or using one source's edges for both, is wrong -- it silently
mixes different bin widths.  This script assigns each source's counts to
magnitude bands using *that source's own* edges and compares **mass fractions**,
which are dimensionless and comparable.

``--check`` recomputes the tail quantile ratios and is expected to reproduce the
run's logged ``histogram/prediction_frac_of_target/<pct>th-percentile/<VAR>``
almost exactly; treat a mismatch as a bug in this script, not a finding.

Fetch the netCDF with::

    beaker dataset fetch <result-dataset-ULID> \
        --prefix evaluator_maps_and_metrics.nc -o <dir>

Usage::

    python diagnose_eval_histogram_spectrum.py <run.nc> [<run2.nc> ...] \
        [--var PRATEsfc] [--check]
"""

import argparse

import numpy as np
import xarray as xr

# kg/m2/s -> mm/day for precipitation rates.
SECONDS_PER_DAY = 86400.0

DEFAULT_BANDS = [(50, 100), (100, 200), (200, 300), (300, 400), (400, 600)]
DEFAULT_WAVENUMBERS = [10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
DEFAULT_PERCENTILES = [99.99, 99.9999]


def _mass_fraction(
    edges: np.ndarray, counts: np.ndarray, lo: float, hi: float
) -> float:
    """Fraction of total mass in [lo, hi), using this source's own edges."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    selected = (centers >= lo) & (centers < hi)
    total = counts.sum()
    return float(counts[selected].sum() / total) if total > 0 else float("nan")


def _exceedance(edges: np.ndarray, counts: np.ndarray, threshold: float) -> float:
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    return (
        float(counts[centers >= threshold].sum() / total) if total > 0 else float("nan")
    )


def _quantile(edges: np.ndarray, counts: np.ndarray, p: float) -> float:
    """Linear-interpolated quantile at fraction ``p`` from a binned histogram."""
    cumulative = np.cumsum(counts)
    target = p * cumulative[-1]
    i = min(int(np.searchsorted(cumulative, target)), len(counts) - 1)
    below = cumulative[i - 1] if i > 0 else 0.0
    frac = (target - below) / counts[i] if counts[i] > 0 else 0.0
    return float(edges[i] + frac * (edges[i + 1] - edges[i]))


def _histogram_arrays(ds: xr.Dataset, var: str, scale: float):
    counts = ds[f"histogram_{var}"]
    edges = ds[f"histogram_{var}_bin_edges"]
    return (
        counts.sel(source="target").values.astype(float),
        counts.sel(source="prediction").values.astype(float),
        edges.sel(source="target").values.astype(float) * scale,
        edges.sel(source="prediction").values.astype(float) * scale,
    )


def report_histogram(
    ds: xr.Dataset, var: str, bands, scale: float, check: bool
) -> None:
    count_t, count_p, edge_t, edge_p = _histogram_arrays(ds, var, scale)
    print(
        f"  histogram bin width: target {edge_t[1] - edge_t[0]:.3f}, "
        f"prediction {edge_p[1] - edge_p[0]:.3f} mm/day "
        f"({'shared' if np.allclose(edge_t, edge_p) else 'INDEPENDENT — use each own'})"
    )
    print(f"  {'band (mm/day)':>16} {'target':>11} {'prediction':>11} {'ratio':>8}")
    for lo, hi in bands:
        f_t = _mass_fraction(edge_t, count_t, lo, hi)
        f_p = _mass_fraction(edge_p, count_p, lo, hi)
        ratio = f_p / f_t if f_t > 0 else float("nan")
        print(
            f"  {f'{lo}-{hi}':>16} {f_t:11.4e} {f_p:11.4e} {ratio:7.3f} "
            f"({100 * (ratio - 1):+6.1f}%)"
        )
    for threshold in (200.0, 400.0):
        e_t = _exceedance(edge_t, count_t, threshold)
        e_p = _exceedance(edge_p, count_p, threshold)
        print(
            f"  exceedance >{threshold:.0f}: target {e_t:.3e}  prediction {e_p:.3e}"
            f"  x{e_p / e_t:.2f}"
        )
    if check:
        print(
            "  quantile check (should match the run's logged "
            "prediction_frac_of_target):"
        )
        for pct in DEFAULT_PERCENTILES:
            q_t = _quantile(edge_t, count_t, pct / 100.0)
            q_p = _quantile(edge_p, count_p, pct / 100.0)
            print(
                f"    @{pct}: target {q_t:8.1f} prediction {q_p:8.1f} mm/day"
                f"  ratio {q_p / q_t:.4f}"
            )


def report_spectrum(ds: xr.Dataset, var: str, wavenumbers) -> None:
    prediction = ds[f"power_spectrum_fine.{var}"].values
    target = ds[f"power_spectrum_target.{var}"].values
    ratio = prediction / target
    n = len(ratio)
    # k/k_max == 2 * pixel_size / wavelength, so the fractional position of a
    # given physical wavelength is grid-width independent. Equal-thirds bands
    # (as used by the training-val spec_mae_{lo,mid,hi}) split at 1/3 and 2/3.
    print(
        f"  {n} wavenumbers; equal-thirds bands split at k={n // 3} and k={2 * n // 3}"
    )
    print(f"  {'k':>6} {'k/k_max':>8} {'ratio':>8}")
    for k in [k for k in wavenumbers if k < n]:
        print(
            f"  {k:6d} {k / (n - 1):8.3f} {ratio[k]:7.3f} "
            f"({100 * (ratio[k] - 1):+6.1f}%)"
        )
    for name, lo, hi in [
        ("lo", 1, n // 3),
        ("mid", n // 3, 2 * n // 3),
        ("hi", 2 * n // 3, n),
    ]:
        band = np.abs(ratio[lo:hi] - 1.0)
        worst = lo + int(np.argmax(band))
        print(
            f"  {name:>3} band k=[{lo},{hi}): max |error| {100 * band.max():5.1f}% "
            f"at k={worst} (k/k_max={worst / (n - 1):.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="evaluator_maps_and_metrics.nc files")
    parser.add_argument("--var", default="PRATEsfc", help="output variable")
    parser.add_argument(
        "--scale",
        type=float,
        default=SECONDS_PER_DAY,
        help="multiplier applied to histogram edges (default kg/m2/s -> mm/day)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute tail quantile ratios to validate against logged metrics",
    )
    args = parser.parse_args()

    for path in args.paths:
        ds = xr.open_dataset(path)
        print(f"=== {path}  ({args.var}) ===")
        report_histogram(ds, args.var, DEFAULT_BANDS, args.scale, args.check)
        print()
        report_spectrum(ds, args.var, DEFAULT_WAVENUMBERS)
        print()


if __name__ == "__main__":
    main()
