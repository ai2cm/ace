"""
Compare spherical power spectra across three ERA5 time periods to assess
how much of the start-vs-end spectral difference is attributable to the
pre-1979 regime versus post-1979 drift.

Periods:
  1. Pre-1979:  1940-1978  (before the training split / regime shift)
  2. Post-1979: 1979-2000
  3. Early 2000s: 2001-2009
  4. Post-2010: 2010-2022  (known ERA5 stream boundary at 2010)

For each period, we sample several years, compute the spherical power
spectrum (same method as fme.core.metrics.spherical_power_spectrum), and
average across samples. We then plot and compare.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# Use the project's own SHT implementation (fork of torch_harmonics)
from fme.sht_fix import RealSHT

ZARR_PATH = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
CACHE_DIR = "/tmp/era5_samples"
os.makedirs(CACHE_DIR, exist_ok=True)

# Variables to analyze — a mix of upper-atmosphere and surface
VARIABLES = [
    "air_temperature_0",
    "air_temperature_7",
    "specific_total_water_0",
    "specific_total_water_7",
    "eastward_wind_0",
    "PRESsfc",
    "TMP2m",
    "PRATEsfc",
]

# Sample years from each period — 12 sparse timesteps per year keeps it fast
PERIODS = {
    "1940-1978 (pre-1979)": [1945, 1950, 1955, 1960, 1965, 1970, 1975],
    "1979-2000 (post-1979)": [1980, 1985, 1990, 1995, 2000],
    "2001-2009": [2002, 2004, 2006, 2008],
    "2010-2022 (post-2010)": [2011, 2014, 2017, 2020],
}

# Number of timesteps to sample per year (spread evenly across the year)
TIMESTEPS_PER_YEAR = 12


def spherical_power_spectrum(field: torch.Tensor, sht: RealSHT) -> torch.Tensor:
    """Compute spherical power spectrum — mirrors fme.core.metrics version."""
    field_sht = sht(field)
    power_spectrum = torch.sum(abs(field_sht) ** 2, dim=-1)
    return power_spectrum


def load_samples_for_year(ds, year, variables):
    """Load a sparse sample of timesteps for one year."""
    cache_file = os.path.join(CACHE_DIR, f"spectral_sample_{year}.nc")
    if os.path.exists(cache_file):
        return xr.open_dataset(cache_file)

    t0 = time.time()
    chunk = ds[variables].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    n_times = chunk.sizes["time"]
    if n_times == 0:
        raise ValueError(f"No data for year {year}")

    # Sample evenly spaced timesteps
    indices = np.linspace(0, n_times - 1, TIMESTEPS_PER_YEAR, dtype=int)
    sample = chunk.isel(time=indices).load()
    elapsed = time.time() - t0
    print(f"  {year}: loaded {len(indices)} timesteps in {elapsed:.1f}s")

    sample.to_netcdf(cache_file)
    return sample


def compute_mean_spectrum_for_period(ds, period_name, years, variables, sht):
    """Compute the mean power spectrum across all sampled timesteps in a period."""
    all_spectra: dict[str, list[torch.Tensor]] = {var: [] for var in variables}

    for year in years:
        try:
            sample = load_samples_for_year(ds, year, variables)
        except Exception as e:
            print(f"  WARNING: skipping {year}: {e}")
            continue

        for var in variables:
            if var not in sample:
                continue
            data = sample[var].values  # (time, lat, lon)
            tensor = torch.from_numpy(data).float().unsqueeze(1)  # (time, 1, lat, lon)
            spectrum = spherical_power_spectrum(tensor, sht)  # (time, 1, lmax)
            mean_spectrum = spectrum.mean(dim=0).squeeze(0)  # (lmax,)
            all_spectra[var].append(mean_spectrum)

    # Average across years
    result = {}
    for var in variables:
        if all_spectra[var]:
            result[var] = torch.stack(all_spectra[var]).mean(dim=0)
    return result


def plot_spectra_comparison(spectra_by_period, variables):
    """Plot power spectra for four periods, one figure per variable."""
    period_names = list(spectra_by_period.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for var in variables:
        fig, ax = plt.subplots(figsize=(7, 5))
        for j, period in enumerate(period_names):
            if var in spectra_by_period[period]:
                spectrum = spectra_by_period[period][var].numpy()
                wavenumbers = np.arange(len(spectrum))
                ax.plot(
                    wavenumbers, spectrum, color=colors[j], label=period, linewidth=1.5
                )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Wavenumber (l)")
        ax.set_ylabel("Power")
        ax.set_title(f"Spherical Power Spectrum: {var}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        fname = os.path.join(PLOT_DIR, f"spectral_power_{var}.png")
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_spectral_ratios(spectra_by_period, variables):
    """Plot ratios of spectra, one figure per variable."""
    period_names = list(spectra_by_period.keys())

    for var in variables:
        has_all = all(var in spectra_by_period[p] for p in period_names)
        if not has_all:
            continue

        s_pre79 = spectra_by_period[period_names[0]][var].numpy()
        s_post79 = spectra_by_period[period_names[1]][var].numpy()
        s_early2000s = spectra_by_period[period_names[2]][var].numpy()
        s_post2010 = spectra_by_period[period_names[3]][var].numpy()
        wavenumbers = np.arange(len(s_pre79))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            wavenumbers[1:],
            (s_post2010 / s_pre79)[1:],
            color="#d62728",
            label="2010-2022 / Pre-1979 (full)",
            linewidth=1.5,
        )
        ax.plot(
            wavenumbers[1:],
            (s_post2010 / s_post79)[1:],
            color="#9467bd",
            label="2010-2022 / 1979-2000",
            linewidth=1.5,
        )
        ax.plot(
            wavenumbers[1:],
            (s_post2010 / s_early2000s)[1:],
            color="#17becf",
            label="2010-2022 / 2001-2009",
            linewidth=1.5,
        )
        ax.plot(
            wavenumbers[1:],
            (s_post79 / s_pre79)[1:],
            color="#bcbd22",
            label="1979-2000 / Pre-1979",
            linewidth=1.2,
            linestyle="--",
        )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Wavenumber (l)")
        ax.set_ylabel("Power Ratio")
        ax.set_title(f"Spectral Power Ratios: {var}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        fname = os.path.join(PLOT_DIR, f"spectral_ratio_{var}.png")
        fig.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def print_summary(spectra_by_period, variables):
    """Print a quantitative summary of spectral differences."""
    period_names = list(spectra_by_period.keys())

    print("\n" + "=" * 110)
    print(
        "QUANTITATIVE SUMMARY: SPECTRAL POWER DIFFERENCES (small scales, top 20% wavenumbers)"
    )
    print("=" * 110)

    # Pairwise comparisons of interest
    comparisons = [
        (period_names[3], period_names[0], "2010-2022 vs Pre-1979 (full)"),
        (period_names[3], period_names[1], "2010-2022 vs 1979-2000"),
        (period_names[3], period_names[2], "2010-2022 vs 2001-2009"),
        (period_names[1], period_names[0], "1979-2000 vs Pre-1979"),
        (period_names[2], period_names[1], "2001-2009 vs 1979-2000"),
    ]

    print(f"\n{'Variable':<28}", end="")
    for _, _, label in comparisons:
        print(f" {label:>20}", end="")
    print()
    print("-" * 130)

    for var in variables:
        has_all = all(var in spectra_by_period[p] for p in period_names)
        if not has_all:
            continue

        print(f"{var:<28}", end="")
        for p_num, p_den, _ in comparisons:
            s_num = spectra_by_period[p_num][var].numpy()
            s_den = spectra_by_period[p_den][var].numpy()
            n = len(s_num)
            ss = slice(int(0.8 * n), n)
            ratio = np.mean(s_num[ss] / s_den[ss])
            print(f" {ratio:>20.4f}x", end="")
        print()

    # RMSE of log-ratio across all wavenumbers
    print(f"\n--- RMSE of log10(ratio) across all wavenumbers (excl. l=0) ---")
    print(f"{'Variable':<28}", end="")
    for _, _, label in comparisons:
        print(f" {label:>20}", end="")
    print()
    print("-" * 130)

    for var in variables:
        has_all = all(var in spectra_by_period[p] for p in period_names)
        if not has_all:
            continue

        print(f"{var:<28}", end="")
        for p_num, p_den, _ in comparisons:
            s_num = spectra_by_period[p_num][var].numpy()
            s_den = spectra_by_period[p_den][var].numpy()
            log_diff = np.log10(s_num[1:]) - np.log10(s_den[1:])
            rmse = np.sqrt(np.mean(log_diff**2))
            print(f" {rmse:>20.6f}", end="")
        print()


def main():
    print("Opening zarr dataset...")
    ds = xr.open_zarr(ZARR_PATH)

    # Determine grid size from the dataset
    nlat = ds.sizes["latitude"]
    nlon = ds.sizes["longitude"]
    print(f"Grid: {nlat} x {nlon}")

    # Build the SHT on CPU
    sht = RealSHT(nlat, nlon, lmax=nlat, mmax=nlat, grid="equiangular")

    print(
        f"\nComputing power spectra for {len(VARIABLES)} variables across 3 periods..."
    )

    spectra_by_period = {}
    for period_name, years in PERIODS.items():
        print(f"\n--- {period_name} (years: {years}) ---")
        spectra_by_period[period_name] = compute_mean_spectrum_for_period(
            ds, period_name, years, VARIABLES, sht
        )

    # Plot
    plot_spectra_comparison(spectra_by_period, VARIABLES)
    plot_spectral_ratios(spectra_by_period, VARIABLES)
    print_summary(spectra_by_period, VARIABLES)


if __name__ == "__main__":
    main()
