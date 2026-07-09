"""
Deep dive into the most surprising features found in the initial analysis.
Uses the already-downloaded 1996 sample.
"""

import numpy as np
import xarray as xr

SAMPLE_FILE = "/tmp/era5_samples/era5_1996_sample.nc"


def load_sample():
    return xr.open_dataset(SAMPLE_FILE)


def analyze_stw1_extreme_distribution(ds):
    """specific_total_water_1 has skewness 190 and kurtosis 124k - investigate why."""
    print("=" * 100)
    print("DEEP DIVE: specific_total_water_1 (97.7 hPa) extreme distribution")
    print("=" * 100)

    data = ds["specific_total_water_1"].values  # (time, lat, lon)
    flat = data.flatten()

    print(f"\nOverall stats: mean={np.mean(flat):.4e}, median={np.median(flat):.4e}")
    print(f"  P50={np.percentile(flat, 50):.4e}, P90={np.percentile(flat, 90):.4e}")
    print(f"  P95={np.percentile(flat, 95):.4e}, P99={np.percentile(flat, 99):.4e}")
    print(
        f"  P99.9={np.percentile(flat, 99.9):.4e}, P99.99={np.percentile(flat, 99.99):.4e}"
    )
    print(f"  Max={np.max(flat):.4e}")

    # What fraction of the variance comes from extreme values?
    total_var = np.var(flat)
    threshold = np.percentile(flat, 99)
    mask_normal = flat <= threshold
    print(f"\nVariance decomposition:")
    print(f"  Total variance: {total_var:.4e}")
    print(f"  Variance excluding top 1%: {np.var(flat[mask_normal]):.4e}")
    print(f"  Ratio: {np.var(flat[mask_normal]) / total_var:.4f}")

    # Spatial distribution of extreme values
    threshold_99p9 = np.percentile(flat, 99.9)
    extreme_mask = data > threshold_99p9
    extreme_by_lat = np.sum(extreme_mask, axis=(0, 2))  # sum over time and lon
    lats = ds.latitude.values
    print(f"\nSpatial distribution of top 0.1% values (> {threshold_99p9:.4e}):")
    for i in range(0, len(lats), 10):
        print(f"  Lat {lats[i]:>6.1f}: {extreme_by_lat[i]:>6d} occurrences")

    # What does the "background" distribution look like?
    background = flat[flat < np.percentile(flat, 95)]
    print(
        f"\nBackground (bottom 95%): mean={np.mean(background):.4e}, std={np.std(background):.4e}"
    )
    print(f"  Range: [{np.min(background):.4e}, {np.max(background):.4e}]")

    # Check: is this bimodal? Tropics vs polar?
    tropical = data[:, 70:110, :]  # roughly -20 to 20 deg
    polar = np.concatenate([data[:, :30, :], data[:, 150:, :]], axis=1)
    print(
        f"\nTropical (20S-20N): mean={np.mean(tropical):.4e}, std={np.std(tropical):.4e}, max={np.max(tropical):.4e}"
    )
    print(
        f"Polar (>60): mean={np.mean(polar):.4e}, std={np.std(polar):.4e}, max={np.max(polar):.4e}"
    )


def analyze_northward_wind_0_autocorr(ds):
    """northward_wind_0 has autocorrelation of only 0.37 - investigate."""
    print("\n" + "=" * 100)
    print("DEEP DIVE: northward_wind_0 (25.6 hPa) low temporal autocorrelation")
    print("=" * 100)

    data = ds["northward_wind_0"].values

    # Global mean time series autocorrelation at various lags
    gm = np.nanmean(data, axis=(1, 2))
    gm_centered = gm - np.mean(gm)
    var = np.var(gm_centered)
    print("\nGlobal mean autocorrelation at various lags:")
    for lag in [1, 2, 4, 8, 16, 32]:
        if lag < len(gm):
            acf = np.mean(gm_centered[:-lag] * gm_centered[lag:]) / var
            print(f"  Lag {lag} ({lag*6}h): {acf:.4f}")

    # Compare with other v-wind levels and u-wind_0
    print("\nComparison of autocorrelation (lag-1) across levels:")
    for var_base in ["northward_wind", "eastward_wind"]:
        for i in range(8):
            name = f"{var_base}_{i}"
            ts = np.nanmean(ds[name].values, axis=(1, 2))
            ts_c = ts - np.mean(ts)
            v = np.var(ts_c)
            acf = np.mean(ts_c[:-1] * ts_c[1:]) / v if v > 0 else 0
            print(f"  {name:<25}: autocorr={acf:.4f}, global_mean_std={np.std(ts):.4f}")

    # Check if it's lat-dependent
    lats = ds.latitude.values
    print("\nLatitudinal autocorrelation for northward_wind_0:")
    for lat_idx in [0, 30, 60, 90, 120, 150, 179]:
        ts = np.nanmean(data[:, lat_idx, :], axis=1)
        ts_c = ts - np.mean(ts)
        v = np.var(ts_c)
        acf = np.mean(ts_c[:-1] * ts_c[1:]) / v if v > 0 else 0
        print(f"  Lat {lats[lat_idx]:>6.1f}: autocorr={acf:.4f}, std={np.std(ts):.4f}")


def analyze_normalization_implications(ds):
    """Understand how the extreme full/residual ratios impact training."""
    print("\n" + "=" * 100)
    print("DEEP DIVE: Normalization and loss implications")
    print("=" * 100)

    scaling_full = xr.open_dataset("/tmp/era5_scaling_full.nc")
    scaling_resid = xr.open_dataset("/tmp/era5_scaling_residual.nc")

    print("\nNormalized residual statistics (what the loss actually sees):")
    print(
        "(residual / full_field_scaling = how large residuals are in the network's normalized space)"
    )
    print(
        f"{'Variable':<50} {'Resid/FullScale':>15} {'Resid/ResidScale':>15} {'FullScale/ResidScale':>20}"
    )
    print("-" * 105)

    vars_to_check = []
    for base in [
        "air_temperature",
        "specific_total_water",
        "eastward_wind",
        "northward_wind",
    ]:
        vars_to_check += [f"{base}_{i}" for i in range(8)]
    vars_to_check += ["PRESsfc", "TMP2m", "Q2m", "PRATEsfc"]

    for var in vars_to_check:
        if var not in ds or var not in scaling_full:
            continue
        data = ds[var].values
        residuals = np.diff(data, axis=0)
        resid_std = np.std(residuals)
        sf = float(scaling_full[var].values)
        sr = float(scaling_resid[var].values)
        print(f"{var:<50} {resid_std/sf:>15.4f} {resid_std/sr:>15.4f} {sf/sr:>20.2f}")


def analyze_signal_to_noise(ds):
    """For each variable, compute signal-to-noise ratio relevant to learning."""
    print("\n" + "=" * 100)
    print("DEEP DIVE: Signal-to-noise ratios for learning")
    print("=" * 100)
    print("Signal = 6-hour change (what model must predict)")
    print(
        "Noise-like = how much the 6-hour change varies spatially within a single timestep"
    )
    print()

    print(
        f"{'Variable':<50} {'Resid Std':>12} {'SpatialVar':>12} {'TemporalVar':>12} {'Temp/Spat':>10}"
    )
    print("-" * 100)

    for base in [
        "air_temperature",
        "specific_total_water",
        "eastward_wind",
        "northward_wind",
    ]:
        for i in range(8):
            name = f"{base}_{i}"
            if name not in ds:
                continue
            data = ds[name].values
            residuals = np.diff(data, axis=0)

            # Temporal variance of residuals (how much the change itself varies over time)
            mean_resid_per_time = np.mean(residuals, axis=(1, 2))  # (time,)
            temporal_var = np.var(mean_resid_per_time)

            # Spatial variance of residuals (how much the change varies across space at a given time)
            spatial_vars = np.var(residuals, axis=(1, 2))  # (time,)
            mean_spatial_var = np.mean(spatial_vars)

            resid_std = np.std(residuals)

            ratio = (
                temporal_var / mean_spatial_var
                if mean_spatial_var > 0
                else float("inf")
            )
            print(
                f"{name:<50} {resid_std:>12.4g} {mean_spatial_var:>12.4g} {temporal_var:>12.4g} {ratio:>10.4f}"
            )


def analyze_stw0_near_constant(ds):
    """specific_total_water_0 is nearly constant - explore implications."""
    print("\n" + "=" * 100)
    print("DEEP DIVE: specific_total_water_0 (25.6 hPa) nearly constant field")
    print("=" * 100)

    data = ds["specific_total_water_0"].values
    print(f"Range: [{np.min(data):.6e}, {np.max(data):.6e}]")
    print(f"Mean: {np.mean(data):.6e}")
    print(f"Std: {np.std(data):.6e}")
    print(f"Coefficient of variation: {np.std(data)/np.mean(data)*100:.2f}%")

    # How much does it vary spatially vs temporally?
    time_mean = np.mean(data, axis=0)  # (lat, lon) - time-averaged field
    spatial_std_of_time_mean = np.std(time_mean)
    residuals = data - time_mean[np.newaxis, :, :]
    temporal_std = np.std(residuals)
    print(f"\nSpatial std of time-mean: {spatial_std_of_time_mean:.6e}")
    print(f"Temporal std (after removing time-mean): {temporal_std:.6e}")

    # After normalization with scaling-full-field, what does this look like?
    scaling_full = xr.open_dataset("/tmp/era5_scaling_full.nc")
    centering = xr.open_dataset("/tmp/era5_centering.nc")
    sf = float(scaling_full["specific_total_water_0"].values)
    center = float(centering["specific_total_water_0"].values)
    normalized = (data - center) / sf
    print(
        f"\nAfter full-field normalization: mean={np.mean(normalized):.4f}, std={np.std(normalized):.4f}"
    )
    print(f"  Range: [{np.min(normalized):.4f}, {np.max(normalized):.4f}]")

    # 6-hour change normalized
    resid = np.diff(normalized, axis=0)
    print(f"  6h change std (in norm space): {np.std(resid):.6f}")
    print(
        f"  This means ~{np.std(resid)*100:.2f}% of full-field std changes per timestep"
    )


def analyze_temperature_inversions(ds):
    """The non-monotonic temperature profile is physically interesting."""
    print("\n" + "=" * 100)
    print("DEEP DIVE: Temperature profile non-monotonicity")
    print("=" * 100)

    levels_hpa = [25.6, 97.7, 199.1, 330.5, 503.5, 689.2, 847.1, 964.4]
    temps = []
    for i in range(8):
        t = float(np.mean(ds[f"air_temperature_{i}"].values))
        temps.append(t)
        print(f"Level {i} ({levels_hpa[i]:>6.1f} hPa): {t:.2f} K")

    # Is this inversion consistent across latitudes?
    lats = ds.latitude.values
    print("\nTemperature at each level by latitude band:")
    print(f"{'Lat band':<15}" + "".join(f"{'L'+str(i):>10}" for i in range(8)))
    for lat_start, lat_end, name in [
        (-90, -60, "60S-90S"),
        (-30, 30, "30S-30N"),
        (60, 90, "60N-90N"),
    ]:
        lat_mask = (lats >= lat_start) & (lats <= lat_end)
        vals = []
        for i in range(8):
            t = float(np.mean(ds[f"air_temperature_{i}"].values[:, lat_mask, :]))
            vals.append(t)
        print(f"{name:<15}" + "".join(f"{v:>10.2f}" for v in vals))

    # Where is the tropopause in terms of our levels?
    print("\nLevel 1 (97.7 hPa) is the tropopause level - coldest point in the profile")
    print(
        "Level 0 (25.6 hPa) is in the stratosphere - warms again due to ozone heating"
    )
    print(
        "This means temperature_0 and temperature_1 have inverted relationship to lower levels"
    )


def analyze_loss_weight_effective_signal(ds):
    """What does the effective loss weighting look like considering normalization?"""
    print("\n" + "=" * 100)
    print("DEEP DIVE: Effective loss weighting analysis")
    print("=" * 100)

    scaling_full = xr.open_dataset("/tmp/era5_scaling_full.nc")

    # Loss weights from config
    loss_weights = {
        "air_temperature_0": 0.5,
        "air_temperature_1": 0.5,
        "eastward_wind_0": 0.5,
        "northward_wind_0": 0.5,
        "specific_total_water_0": 0.5,
        "specific_total_water_1": 0.25,
        "specific_total_water_2": 0.5,
        "PRATEsfc": 0.5,
        "h500": 10,
        "TMP850": 5,
        "Q2m": 0.5,
        "DLWRFsfc": 2,
        "ULWRFsfc": 5,
        "USWRFsfc": 2,
        "DSWRFsfc": 2,
        "USWRFtoa": 2,
        "tendency_of_total_water_path_due_to_advection": 0.5,
        "PRESsfc": 3,
    }

    # The loss operates on normalized predictions, but what physical error does a unit
    # loss correspond to?
    # Loss is computed on (prediction - target) / scaling
    # So loss = 1 means error = 1 * scaling
    print(
        f"{'Variable':<50} {'Weight':>8} {'Scale':>12} {'Weight*Scale^-2':>15} {'Resid Std':>12} {'Eff SNR':>10}"
    )
    print("-" * 110)

    for var, weight in sorted(loss_weights.items()):
        if var not in scaling_full or var not in ds:
            continue
        sf = float(scaling_full[var].values)
        data = ds[var].values
        if len(data.shape) == 3:
            resid_std = np.std(np.diff(data, axis=0))
        else:
            resid_std = 0

        # Effective sensitivity: weight / scale^2 determines how much a physical unit error matters
        eff = weight / (sf**2)
        # Effective SNR: how big is the residual signal relative to the normalized scale
        snr = resid_std / sf if sf > 0 else 0
        print(
            f"{var:<50} {weight:>8.2f} {sf:>12.4g} {eff:>15.4g} {resid_std:>12.4g} {snr:>10.4f}"
        )


def analyze_radiation_bimodality(ds):
    """Solar radiation is zero at night - bimodal distribution."""
    print("\n" + "=" * 100)
    print("DEEP DIVE: Radiation variable bimodality and zero-inflation")
    print("=" * 100)

    for var in ["DSWRFsfc", "USWRFsfc", "USWRFtoa", "PRATEsfc"]:
        data = ds[var].values.flatten()
        n_total = len(data)
        n_zero = np.sum(data == 0)
        n_neg = np.sum(data < 0)
        nonzero = data[data > 0]
        print(f"\n{var}:")
        print(f"  Total points: {n_total}")
        print(f"  Exact zeros: {n_zero} ({100*n_zero/n_total:.1f}%)")
        print(f"  Negative: {n_neg} ({100*n_neg/n_total:.4f}%)")
        if len(nonzero) > 0:
            print(
                f"  Non-zero stats: mean={np.mean(nonzero):.4g}, std={np.std(nonzero):.4g}"
            )
            print(
                f"  Non-zero percentiles: P1={np.percentile(nonzero, 1):.4g}, P50={np.percentile(nonzero, 50):.4g}, P99={np.percentile(nonzero, 99):.4g}"
            )

        # Residuals are huge for radiation due to day/night cycle
        data3d = ds[var].values
        resid = np.diff(data3d, axis=0)
        print(f"  Residual std: {np.std(resid):.4g}")
        print(f"  Residual range: [{np.min(resid):.4g}, {np.max(resid):.4g}]")

        # What fraction of the residual variance comes from the diurnal cycle?
        # Approximate: how much does the field change at 6-hour intervals?
        gm_series = np.mean(data3d, axis=(1, 2))  # global mean time series
        gm_resid = np.diff(gm_series)
        print(f"  Global-mean residual std: {np.std(gm_resid):.4g}")


if __name__ == "__main__":
    ds = load_sample()
    analyze_stw1_extreme_distribution(ds)
    analyze_northward_wind_0_autocorr(ds)
    analyze_normalization_implications(ds)
    analyze_signal_to_noise(ds)
    analyze_stw0_near_constant(ds)
    analyze_temperature_inversions(ds)
    analyze_loss_weight_effective_signal(ds)
    analyze_radiation_bimodality(ds)
