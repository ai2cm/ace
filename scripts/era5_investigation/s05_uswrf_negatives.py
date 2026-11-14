"""
Investigate why USWRFsfc has small negative values while DSWRFsfc and USWRFtoa don't.

USWRFsfc = upwelling shortwave radiation at the surface (reflected solar).
Physically this should be non-negative (you can't have negative reflected sunlight).
"""

import numpy as np
import xarray as xr

ZARR_PATH = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"
CACHE_FILE = "/tmp/era5_samples/era5_1996_sample.nc"

# Load a sample - the cached file should have radiation variables
print("Loading cached sample data...")
ds = xr.open_dataset(CACHE_FILE)
print("Variables in cache:", list(ds.data_vars))

rad_vars = [v for v in ["USWRFsfc", "DSWRFsfc", "USWRFtoa", "DLWRFsfc"] if v in ds]
print(f"\nRadiation variables available: {rad_vars}")

if "USWRFsfc" in ds:
    usw = ds["USWRFsfc"].values
    print(f"\n{'='*60}")
    print("USWRFsfc detailed statistics")
    print("=" * 60)
    print(f"  Shape: {usw.shape}")
    print(f"  Min: {np.nanmin(usw):.6f}")
    print(f"  Max: {np.nanmax(usw):.6f}")
    print(f"  Mean: {np.nanmean(usw):.6f}")

    neg_mask = usw < 0
    n_neg = np.sum(neg_mask)
    n_total = usw.size
    print(f"\n  Negative values: {n_neg} / {n_total} ({100*n_neg/n_total:.2f}%)")

    if n_neg > 0:
        neg_vals = usw[neg_mask]
        print(
            f"  Negative value range: [{np.min(neg_vals):.6f}, {np.max(neg_vals):.6f}]"
        )
        print(f"  Negative value mean: {np.mean(neg_vals):.6f}")
        print(f"  Negative value median: {np.median(neg_vals):.6f}")

        # Percentiles of negative values
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentiles of negative values:")
        for p in pcts:
            print(f"    {p}th: {np.percentile(neg_vals, p):.6f}")

        # Are negatives correlated with nighttime (DSWRFsfc == 0)?
        if "DSWRFsfc" in ds:
            dsw = ds["DSWRFsfc"].values
            nighttime = dsw == 0
            neg_and_night = neg_mask & nighttime
            neg_and_day = neg_mask & ~nighttime
            print(f"\n  Nighttime (DSWRFsfc=0): {np.sum(nighttime)} points")
            print(f"  Negatives during nighttime: {np.sum(neg_and_night)}")
            print(f"  Negatives during daytime (DSWRFsfc>0): {np.sum(neg_and_day)}")

            # What are DSWRFsfc values where USWRFsfc is negative?
            dsw_at_neg = dsw[neg_mask]
            print(f"\n  DSWRFsfc where USWRFsfc<0:")
            print(f"    Mean: {np.mean(dsw_at_neg):.4f}")
            print(f"    Median: {np.median(dsw_at_neg):.4f}")
            print(f"    Min: {np.min(dsw_at_neg):.4f}")
            print(f"    Max: {np.max(dsw_at_neg):.4f}")
            print(f"    % that are zero: {100*np.mean(dsw_at_neg == 0):.1f}%")

        # Spatial distribution of negatives
        if usw.ndim >= 2:
            # Assuming dims are (time, lat, lon) or similar
            # Sum negatives over time to see spatial pattern
            neg_count_spatial = np.sum(neg_mask, axis=0)
            print(f"\n  Spatial distribution of negatives:")
            print(f"    Shape of spatial field: {neg_count_spatial.shape}")
            if neg_count_spatial.ndim == 2:
                # Average by latitude band
                neg_by_lat = np.mean(neg_count_spatial, axis=-1)
                lat_size = neg_by_lat.shape[0]
                print(f"    Negatives by latitude (avg count across longitudes):")
                lat_indices = np.linspace(0, lat_size - 1, 12, dtype=int)
                lats = np.linspace(90, -90, lat_size)
                for li in lat_indices:
                    print(
                        f"      lat~{lats[li]:>6.1f}°: {neg_by_lat[li]:.1f} negative timesteps"
                    )

        # What fraction of negatives are very small?
        print(f"\n  Magnitude analysis:")
        thresholds = [0, -0.01, -0.1, -1.0, -5.0, -10.0]
        for t in thresholds:
            count = np.sum(usw < t)
            print(f"    USWRFsfc < {t:>6.2f}: {count:>8d} ({100*count/n_total:.4f}%)")

    # Compare with other radiation variables
    print(f"\n{'='*60}")
    print("Comparison across radiation variables")
    print("=" * 60)
    for var in rad_vars:
        vals = ds[var].values
        n_neg = np.sum(vals < 0)
        n_zero = np.sum(vals == 0)
        print(f"\n  {var}:")
        print(f"    Range: [{np.nanmin(vals):.4f}, {np.nanmax(vals):.4f}]")
        print(f"    Negatives: {n_neg} ({100*n_neg/vals.size:.2f}%)")
        print(f"    Exact zeros: {n_zero} ({100*n_zero/vals.size:.2f}%)")
        if n_neg > 0:
            print(f"    Most negative: {np.min(vals[vals < 0]):.6f}")
else:
    print("\nUSWRFsfc not in cached sample. Need to load from zarr.")
    print("Loading a small sample from zarr...")
    ds_zarr = xr.open_zarr(ZARR_PATH)
    # Load just radiation vars for a few timesteps
    rad_sample = (
        ds_zarr[["USWRFsfc", "DSWRFsfc", "USWRFtoa"]].isel(time=slice(0, 100)).load()
    )

    for var in ["USWRFsfc", "DSWRFsfc", "USWRFtoa"]:
        vals = rad_sample[var].values
        n_neg = np.sum(vals < 0)
        n_zero = np.sum(vals == 0)
        print(f"\n  {var}:")
        print(f"    Range: [{np.nanmin(vals):.6f}, {np.nanmax(vals):.6f}]")
        print(f"    Negatives: {n_neg} ({100*n_neg/vals.size:.2f}%)")
        print(f"    Exact zeros: {n_zero} ({100*n_zero/vals.size:.2f}%)")
