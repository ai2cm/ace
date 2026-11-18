"""
Download a small sample of ERA5 data and compute detailed statistics.
Focus on properties that could impact ML training, especially upper atmosphere.
"""

import os
import time

import numpy as np
import xarray as xr

ZARR_PATH = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
SAMPLE_DIR = "/tmp/era5_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Variables used in training
LAYERED_VARS = [
    "air_temperature",
    "specific_total_water",
    "eastward_wind",
    "northward_wind",
]
SURFACE_VARS = ["PRESsfc", "TMP2m", "Q2m", "UGRD10m", "VGRD10m", "surface_temperature"]
FLUX_VARS = ["DLWRFsfc", "ULWRFsfc", "DSWRFsfc", "USWRFsfc", "USWRFtoa", "ULWRFtoa"]
DIAG_VARS = [
    "PRATEsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "tendency_of_total_water_path_due_to_advection",
    "TMP850",
    "h500",
]

ALL_VARS = []
for base in LAYERED_VARS:
    ALL_VARS += [f"{base}_{i}" for i in range(8)]
ALL_VARS += SURFACE_VARS + FLUX_VARS + DIAG_VARS


def download_sample():
    """Download a 1-year sample (1996) for analysis."""
    sample_file = os.path.join(SAMPLE_DIR, "era5_1996_sample.nc")
    if os.path.exists(sample_file):
        print(f"Sample already exists at {sample_file}")
        return xr.open_dataset(sample_file)

    print("Opening zarr...")
    t0 = time.time()
    ds = xr.open_zarr(ZARR_PATH)
    print(f"Opened in {time.time()-t0:.1f}s")

    # Select 1996 (validation year in config) and only training variables
    avail_vars = [v for v in ALL_VARS if v in ds]
    print(f"Selecting {len(avail_vars)} variables for 1996...")
    sample = ds[avail_vars].sel(time=slice("1996-01-01", "1996-12-31"))
    print(f"Sample has {sample.sizes['time']} timesteps")

    print("Loading into memory (this may take a while)...")
    t0 = time.time()
    sample = sample.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    sample.to_netcdf(sample_file)
    print(f"Saved to {sample_file}")
    return sample


def compute_basic_stats(ds):
    """Compute per-variable statistics."""
    print("\n" + "=" * 120)
    print("BASIC STATISTICS")
    print("=" * 120)
    header = f"{'Variable':<50} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Skew':>8} {'Kurt':>8} {'%Zero':>8}"
    print(header)
    print("-" * 120)

    for var in ALL_VARS:
        if var not in ds:
            continue
        data = ds[var].values.flatten()
        data = data[~np.isnan(data)]
        if len(data) == 0:
            continue
        mean = np.mean(data)
        std = np.std(data)
        vmin = np.min(data)
        vmax = np.max(data)
        # Skewness and kurtosis
        if std > 0:
            z = (data - mean) / std
            skew = np.mean(z**3)
            kurt = np.mean(z**4) - 3  # excess kurtosis
        else:
            skew = kurt = 0
        pct_zero = 100.0 * np.sum(data == 0) / len(data)
        print(
            f"{var:<50} {mean:>12.4g} {std:>12.4g} {vmin:>12.4g} {vmax:>12.4g} {skew:>8.2f} {kurt:>8.2f} {pct_zero:>7.2f}%"
        )


def analyze_residuals(ds):
    """Analyze 6-hour timestep residuals (what the model actually predicts)."""
    print("\n" + "=" * 120)
    print("RESIDUAL (6-HOUR TENDENCY) STATISTICS")
    print("=" * 120)
    header = f"{'Variable':<50} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Skew':>8} {'Kurt':>8}"
    print(header)
    print("-" * 120)

    for var in ALL_VARS:
        if var not in ds:
            continue
        data = ds[var].values
        residuals = np.diff(data, axis=0)  # along time axis
        residuals = residuals.flatten()
        residuals = residuals[~np.isnan(residuals)]
        if len(residuals) == 0:
            continue
        mean = np.mean(residuals)
        std = np.std(residuals)
        vmin = np.min(residuals)
        vmax = np.max(residuals)
        if std > 0:
            z = (residuals - mean) / std
            skew = np.mean(z**3)
            kurt = np.mean(z**4) - 3
        else:
            skew = kurt = 0
        print(
            f"{var:<50} {mean:>12.4g} {std:>12.4g} {vmin:>12.4g} {vmax:>12.4g} {skew:>8.2f} {kurt:>8.2f}"
        )


def analyze_upper_atmosphere(ds):
    """Deep dive into upper atmosphere variables."""
    print("\n" + "=" * 120)
    print("UPPER ATMOSPHERE DEEP DIVE")
    print("=" * 120)

    pressure_levels = [25.6, 97.7, 199.1, 330.5, 503.5, 689.2, 847.1, 964.4]

    for base in LAYERED_VARS:
        print(f"\n--- {base} ---")
        print(
            f"{'Level':<8} {'P(hPa)':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Resid Std':>12} {'Autocorr':>10} {'Skew':>8} {'Kurt':>8}"
        )
        for i in range(8):
            name = f"{base}_{i}"
            if name not in ds:
                continue
            data = ds[name].values
            flat = data.flatten()
            flat = flat[~np.isnan(flat)]

            mean = np.mean(flat)
            std = np.std(flat)
            vmin = np.min(flat)
            vmax = np.max(flat)

            # Residual std
            residuals = np.diff(data, axis=0).flatten()
            resid_std = np.std(residuals)

            # Temporal autocorrelation (global mean time series)
            ts = np.nanmean(data, axis=(1, 2))
            if len(ts) > 1:
                ts_centered = ts - np.mean(ts)
                autocorr = np.corrcoef(ts_centered[:-1], ts_centered[1:])[0, 1]
            else:
                autocorr = 0

            if std > 0:
                z = (flat - mean) / std
                skew = np.mean(z**3)
                kurt = np.mean(z**4) - 3
            else:
                skew = kurt = 0

            print(
                f"  {i:<6} {pressure_levels[i]:>8.1f} {mean:>12.4g} {std:>12.4g} {vmin:>12.4g} {vmax:>12.4g} {resid_std:>12.4g} {autocorr:>10.6f} {skew:>8.2f} {kurt:>8.2f}"
            )


def analyze_specific_humidity_precision(ds):
    """Check if specific_total_water_0 has float32 precision issues."""
    print("\n" + "=" * 120)
    print("SPECIFIC TOTAL WATER PRECISION ANALYSIS")
    print("=" * 120)

    for i in range(8):
        name = f"specific_total_water_{i}"
        if name not in ds:
            continue
        data = ds[name].values.flatten()
        data = data[~np.isnan(data)]

        # Check dynamic range
        nonzero = data[data > 0]
        if len(nonzero) > 0:
            min_nonzero = np.min(nonzero)
            max_val = np.max(data)
            dynamic_range = max_val / min_nonzero if min_nonzero > 0 else float("inf")
        else:
            min_nonzero = 0
            dynamic_range = 0

        # Number of negative values
        n_neg = np.sum(data < 0)
        n_zero = np.sum(data == 0)

        # Check unique values in smallest magnitude data (precision issues)
        sorted_data = np.sort(data)
        smallest_1pct = sorted_data[: max(1, len(sorted_data) // 100)]
        n_unique_small = len(np.unique(smallest_1pct))

        # Float32 relative precision check
        # float32 has ~7 decimal digits of precision
        mean_val = np.mean(data)
        f32_eps_at_mean = np.float32(mean_val) * np.finfo(np.float32).eps

        print(f"\n{name}:")
        print(f"  Range: [{np.min(data):.4e}, {np.max(data):.4e}]")
        print(f"  Mean: {np.mean(data):.4e}, Std: {np.std(data):.4e}")
        print(f"  Min nonzero: {min_nonzero:.4e}, Dynamic range: {dynamic_range:.1f}")
        print(f"  Negative values: {n_neg} ({100*n_neg/len(data):.4f}%)")
        print(f"  Zero values: {n_zero} ({100*n_zero/len(data):.4f}%)")
        print(f"  Unique values in bottom 1%: {n_unique_small}")
        print(f"  Float32 eps at mean: {f32_eps_at_mean:.4e}")
        print(f"  Residual std: {np.std(np.diff(ds[name].values, axis=0)):.4e}")
        print(
            f"  Ratio residual_std / f32_eps_at_mean: {np.std(np.diff(ds[name].values, axis=0)) / f32_eps_at_mean:.1f}"
        )


def analyze_temporal_jumps(ds):
    """Look for large temporal jumps that could indicate data quality issues."""
    print("\n" + "=" * 120)
    print("TEMPORAL JUMP ANALYSIS (max |residual| / residual_std)")
    print("=" * 120)

    for var in ALL_VARS:
        if var not in ds:
            continue
        data = ds[var].values
        residuals = np.diff(data, axis=0)

        # Global mean residuals
        gm_resid = np.nanmean(residuals, axis=(1, 2))
        gm_std = np.std(gm_resid)
        if gm_std > 0:
            max_jump_idx = np.argmax(np.abs(gm_resid))
            max_jump = gm_resid[max_jump_idx]
            max_jump_sigma = max_jump / gm_std
            jump_time = ds.time.values[max_jump_idx + 1]
            if abs(max_jump_sigma) > 5:
                print(
                    f"{var:<50} max jump: {max_jump_sigma:>8.1f}σ at {str(jump_time)[:19]}"
                )


def analyze_spatial_patterns(ds):
    """Analyze latitudinal distribution patterns."""
    print("\n" + "=" * 120)
    print("LATITUDINAL VARIATION (time-mean, zonal-mean std)")
    print("=" * 120)

    lats = ds.latitude.values

    for base in LAYERED_VARS:
        print(f"\n--- {base}: zonal-mean std at selected latitudes ---")
        print(
            f"{'Level':<8} {'90S':>10} {'60S':>10} {'30S':>10} {'EQ':>10} {'30N':>10} {'60N':>10} {'90N':>10}"
        )
        target_lats = [-89, -60, -30, 0, 30, 60, 89]
        for i in range(8):
            name = f"{base}_{i}"
            if name not in ds:
                continue
            # Time std at each grid point, then zonal mean
            time_std = ds[name].std(dim="time").values  # (lat, lon)
            zonal_mean_std = np.mean(time_std, axis=1)  # (lat,)
            vals = []
            for tlat in target_lats:
                idx = np.argmin(np.abs(lats - tlat))
                vals.append(zonal_mean_std[idx])
            print(f"  {i:<6} " + " ".join(f"{v:>10.4g}" for v in vals))


def analyze_distributions_tails(ds):
    """Check for heavy tails and extreme values."""
    print("\n" + "=" * 120)
    print("DISTRIBUTION TAIL ANALYSIS")
    print("=" * 120)
    print(
        f"{'Variable':<50} {'P0.01':>12} {'P0.1':>12} {'P1':>12} {'P99':>12} {'P99.9':>12} {'P99.99':>12} {'Max':>12}"
    )
    print("-" * 120)

    for var in ALL_VARS:
        if var not in ds:
            continue
        data = ds[var].values.flatten()
        data = data[~np.isnan(data)]
        pcts = np.percentile(data, [0.01, 0.1, 1, 99, 99.9, 99.99])
        vmax = np.max(data)
        print(
            f"{var:<50} {pcts[0]:>12.4g} {pcts[1]:>12.4g} {pcts[2]:>12.4g} {pcts[3]:>12.4g} {pcts[4]:>12.4g} {pcts[5]:>12.4g} {vmax:>12.4g}"
        )


if __name__ == "__main__":
    ds = download_sample()
    compute_basic_stats(ds)
    analyze_residuals(ds)
    analyze_upper_atmosphere(ds)
    analyze_specific_humidity_precision(ds)
    analyze_temporal_jumps(ds)
    analyze_spatial_patterns(ds)
    analyze_distributions_tails(ds)
