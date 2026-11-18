"""
Check for multi-year data discontinuities and trends.
Downloads only global means to keep data transfer minimal.
"""

import os
import time

import numpy as np
import xarray as xr

ZARR_PATH = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"

KEY_VARS = [
    "air_temperature_0",
    "air_temperature_1",
    "air_temperature_7",
    "specific_total_water_0",
    "specific_total_water_1",
    "specific_total_water_7",
    "eastward_wind_0",
    "northward_wind_0",
    "PRESsfc",
    "TMP2m",
    "USWRFsfc",
]

CACHE_FILE = "/tmp/era5_samples/yearly_stats.nc"


def compute_yearly_stats():
    """Compute annual global-mean statistics by loading one year at a time."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached yearly stats from {CACHE_FILE}")
        return xr.open_dataset(CACHE_FILE)

    print("Opening zarr...")
    ds = xr.open_zarr(ZARR_PATH)

    years = range(1940, 2023)
    results: dict[str, list] = {}

    for year in years:
        t0 = time.time()
        try:
            chunk = ds[KEY_VARS].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            n_times = chunk.sizes["time"]
            if n_times == 0:
                continue

            # Only load the global mean to minimize data transfer
            gm = chunk.mean(dim=["latitude", "longitude"]).load()
            elapsed = time.time() - t0

            for var in KEY_VARS:
                ts = gm[var].values
                key_mean = f"{var}_mean"
                key_std = f"{var}_std"
                key_min = f"{var}_min"
                key_max = f"{var}_max"
                results.setdefault(key_mean, []).append(float(np.mean(ts)))
                results.setdefault(key_std, []).append(float(np.std(ts)))
                results.setdefault(key_min, []).append(float(np.min(ts)))
                results.setdefault(key_max, []).append(float(np.max(ts)))

            results.setdefault("year", []).append(year)
            results.setdefault("n_times", []).append(n_times)
            print(f"  {year}: {n_times} timesteps, {elapsed:.1f}s")
        except Exception as e:
            print(f"  {year}: ERROR - {e}")

    # Save as netcdf
    out = xr.Dataset({k: ("year_idx", v) for k, v in results.items()})
    out.to_netcdf(CACHE_FILE)
    print(f"Saved to {CACHE_FILE}")
    return out


def analyze_trends_and_jumps(stats):
    """Look for trends and discontinuities."""
    years = stats["year"].values
    n_times = stats["n_times"].values

    print("\n" + "=" * 100)
    print("MULTI-YEAR ANALYSIS")
    print("=" * 100)

    # Data availability
    print("\nData availability:")
    print(f"  Years: {years[0]} - {years[-1]}")
    print(f"  Total years: {len(years)}")
    # Check for gaps
    for i in range(1, len(years)):
        if years[i] - years[i - 1] > 1:
            print(f"  GAP: {years[i-1]+1}-{years[i]-1}")
    print(f"  Timesteps per year: min={np.min(n_times)}, max={np.max(n_times)}")

    # For each variable, look for jumps and trends
    print("\n--- Year-over-year jumps in global mean ---")
    for var in KEY_VARS:
        means = stats[f"{var}_mean"].values
        # Year-over-year changes
        diffs = np.diff(means)
        diff_std = np.std(diffs)
        if diff_std > 0:
            normalized_diffs = diffs / diff_std
            max_jump_idx = np.argmax(np.abs(normalized_diffs))
            max_jump = normalized_diffs[max_jump_idx]
            if abs(max_jump) > 2.5:
                print(
                    f"  {var:<30}: {max_jump:>6.1f}σ jump from {years[max_jump_idx]} to {years[max_jump_idx+1]} "
                    f"({means[max_jump_idx]:.4g} -> {means[max_jump_idx+1]:.4g})"
                )

    # Known ERA5 discontinuity dates
    known_discontinuities = [
        (1986, "April 1 - parallel stream merge"),
        (1993, "Aug 1 - parallel stream merge"),
        (2000, "Jan 1 - parallel stream merge"),
        (2010, "Jan 1 - parallel stream merge"),
        (2015, "Jan 1 - parallel stream merge"),
        (2019, "Mar 1 - parallel stream merge"),
    ]
    print("\n--- Statistics around known ERA5 discontinuities ---")
    for disc_year, desc in known_discontinuities:
        idx_before = np.where(years == disc_year - 1)[0]
        idx_after = np.where(years == disc_year)[0]
        if len(idx_before) == 0 or len(idx_after) == 0:
            continue
        idx_b = idx_before[0]
        idx_a = idx_after[0]
        print(f"\n  {disc_year} ({desc}):")
        for var in KEY_VARS:
            means = stats[f"{var}_mean"].values
            stds = stats[f"{var}_std"].values
            change_mean = means[idx_a] - means[idx_b]
            change_std = stds[idx_a] - stds[idx_b]
            # Relative to typical year-over-year change
            all_diffs = np.diff(means)
            typical = np.std(all_diffs) if np.std(all_diffs) > 0 else 1
            sigma = change_mean / typical
            if abs(sigma) > 1.5:
                print(
                    f"    {var:<30}: mean change {change_mean:>10.4g} ({sigma:>5.1f}σ), "
                    f"std change {change_std:>10.4g}"
                )

    # Long-term trends
    print("\n--- Long-term trends (1979-2022) ---")
    mask_1979_2022 = (years >= 1979) & (years <= 2022)
    years_sub = years[mask_1979_2022]
    for var in KEY_VARS:
        means = stats[f"{var}_mean"].values[mask_1979_2022]
        if len(means) < 10:
            continue
        # Linear trend
        coeffs = np.polyfit(years_sub, means, 1)
        trend_per_decade = coeffs[0] * 10
        mean_val = np.mean(means)
        pct_per_decade = 100 * trend_per_decade / abs(mean_val) if mean_val != 0 else 0
        print(
            f"  {var:<30}: trend = {trend_per_decade:>12.4g} per decade ({pct_per_decade:>6.2f}%/decade)"
        )


if __name__ == "__main__":
    stats = compute_yearly_stats()
    analyze_trends_and_jumps(stats)
