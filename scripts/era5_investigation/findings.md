# ERA5 Training Data Statistical Analysis

## Introduction

This analysis investigates statistical properties of the ERA5 reanalysis dataset used to pretrain ACE2 models, as configured in `configs/experiments/2025-11-15-ace2s-x-shield/pretrain-1-step-era5.yaml`. The goal is to identify surprising features of the training data that might explain why certain variables (particularly upper atmospheric ones) are difficult to learn, and to suggest changes to training strategy or data handling.

The dataset is a 1-degree, 8-layer vertical representation of ERA5 spanning 1940-2022, stored as zarr at `gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr` (processing config: `scripts/data_process/configs/era5-1deg-8layer-1940-2022.yaml`). It contains 121,262 6-hourly timesteps on a 180x360 lat-lon grid. The 8 vertical levels use hybrid sigma-pressure coordinates with mid-level pressures:

| Level | Pressure (hPa) | Region |
|-------|----------------|--------|
| 0 | 25.6 | Stratosphere |
| 1 | 97.7 | Tropopause |
| 2 | 199.1 | Upper troposphere |
| 3 | 330.5 | Mid troposphere |
| 4 | 503.5 | Mid troposphere |
| 5 | 689.2 | Lower troposphere |
| 6 | 847.1 | Lower troposphere |
| 7 | 964.4 | Near surface |

The training configuration uses:
- `residual_prediction: false` (the network predicts the full field, not the delta)
- Network normalization with `scaling-full-field.nc` (z-score normalization)
- Loss normalization with `scaling-residual.nc` for prognostic variables
- EnsembleLoss combining CRPS (weight 0.9) and energy score (weight 0.1)
- Normalization stats computed over 1990-2019

Analysis scripts are in this directory; cached intermediate data is at `/tmp/era5_samples/`.

---

## Finding 1: specific_total_water_1 (97.7 hPa) has a pathological distribution

This is the single most extreme statistical anomaly in the training data. The distribution has **skewness = 190** and **kurtosis = 124,000**.

- Background (bottom 95% of values): mean = 2.8e-6 kg/kg, std = 4.7e-7
- But extreme values reach 3.8e-3 kg/kg (**1000x the mean**)
- **95.4% of the total variance** comes from the top 1% of values
- Extreme values are concentrated in the tropics (10S-20N), caused by deep convective overshooting injecting moisture into the lower stratosphere
- Temporal autocorrelation is only **0.80** (vs 0.99+ for most variables)
- The residual (6h change) has kurtosis of **130,290**

The normalization scaling (`scaling-full-field.nc`) for this variable is 2.9e-6, which is set by the overall std including the extreme tail. This means for 99% of the data, the normalized values span a very narrow range, and the loss signal is dominated by the rare convective events.

The full-field/residual scaling ratio is **0.86** -- the residual std actually *exceeds* the full-field std, which is unique among the layered variables and reflects how episodic the convective injection events are.

**Implication**: Standard Gaussian normalization is poorly suited. A log-transform, mixture model, or separate handling of background vs extreme convective events could improve learning for this variable. At minimum, the loss gradient is heavily dominated by rare tropical events, which may distort learning for other regions.

---

## Finding 2: northward_wind_0 (25.6 hPa) is nearly unpredictable at 6h resolution

The stratospheric meridional wind has catastrophically low temporal autocorrelation:

| Location | Lag-1 (6h) autocorrelation |
|----------|---------------------------|
| Global mean | 0.37 |
| 60S | 0.86 |
| 30S | 0.32 |
| Equator | 0.63 |
| 30N | 0.41 |
| 90N (pole) | 0.02 |

For comparison, eastward_wind_0 at the same level has autocorrelation **0.999**. The difference is physical: the stratospheric circulation (QBO, polar vortex) is predominantly zonal, so the meridional component has very little coherent temporal structure at 6-hour timescales. The global mean std is only 0.062 m/s (essentially zero coherent global signal), and the kurtosis is 10.79 (very heavy tails -- extreme localized events but no persistent pattern).

Additionally, the autocorrelation does not decay smoothly -- it shows a semi-diurnal signal (lag-4 = 0.57 > lag-2 = 0.32), suggesting some tidal component, but the overall predictability is very low.

**Implication**: The model is essentially being asked to predict noise for this variable. Loss on northward_wind_0 likely adds noisy gradients without meaningful learning signal. Consider reducing its loss weight (currently implicit 1.0, since it's not in the explicit weight list) or removing it from outputs entirely. If it must remain, the loss weight should reflect its low predictability.

---

## Finding 3: Upper atmosphere normalization mismatch

For variables where the full-field/residual scaling ratio is very high, the model must be extremely precise in normalized space to produce meaningful predictions:

| Variable | Full/Resid Ratio | 6h Residual as % of Full Scale |
|----------|-----------------|-------------------------------|
| PRESsfc | 37.8 | 2.6% |
| air_temperature_0 | 19.5 | 5.3% |
| air_temperature_1 | 18.5 | 5.6% |
| eastward_wind_0 | 14.1 | 7.4% |
| air_temperature_3 | 12.5 | 7.9% |
| specific_total_water_0 | 11.9 | 8.8% |

Since `residual_prediction: false`, the network predicts the full field directly. For air_temperature_0, a **5% error in normalized full-field space corresponds to the entire 6-hour residual signal**. The network must reproduce the spatial pattern to ~95% accuracy just to get the tendencies right.

By contrast, lower troposphere and surface variables have much more favorable ratios:

| Variable | Full/Resid Ratio | 6h Residual as % of Full Scale |
|----------|-----------------|-------------------------------|
| northward_wind_7 | 1.9 | 52% |
| UGRD10m | 2.5 | 40% |
| eastward_wind_7 | 2.5 | 39% |
| specific_total_water_1 | 0.86 | 109% |

**Implication**: This is likely a major contributor to why upper atmosphere variables are hard to learn. Switching to `residual_prediction: true` would directly address this by having the network predict the small delta rather than the large full field. Alternatively, level-specific or variable-specific normalization strategies could help.

---

## Finding 4: ERA5 data discontinuities affect stratospheric training data

ERA5 has known discontinuities where parallel data streams were merged (documented in Hersbach 2020). The training config already excludes some problematic periods (1996-2010, 2020) but **includes 1979-1995**, which spans two significant discontinuities:

| Year | Variable most affected | Jump magnitude |
|------|----------------------|---------------|
| 1986 | specific_total_water_0 | -3.9 sigma |
| 1993 | air_temperature_0 | -3.1 sigma |
| 1993 | air_temperature_1 | -2.5 sigma |
| 2000 | specific_total_water_0 | -2.5 sigma |

The 1986 jump in stratospheric water vapor and the 1993 jump in stratospheric temperature are the largest year-over-year discontinuities in the entire 83-year record for those variables. These are reanalysis artifacts, not real climate signals.

**Implication**: Consider excluding 1985-1987 and 1992-1994 from training, or at minimum from stratospheric variable computation. The current training windows include these periods for the 1979-1995 segment.

---

## Finding 5: Radiation variables are bimodal and zero-inflated

| Variable | % Exact Zero | % Negative | Residual Std / Full Std |
|----------|-------------|-----------|------------------------|
| DSWRFsfc | 29.1% | 0% | 1.28 |
| USWRFtoa | 29.0% | 0% | 1.13 |
| USWRFsfc | 0.8% | 23.1% | 0.82 |
| PRATEsfc | 17.9% | 0% | 0.97 |

Key observations:
- DSWRFsfc and USWRFtoa are exactly zero during nighttime (~29% of all data)
- The residual std exceeds the full-field std (ratio > 1) because the diurnal cycle creates transitions of ~1000 W/m2 between day and night
- USWRFsfc has 23% negative values, but they're tiny (~-0.00002 W/m2, float precision noise). The model config correctly forces positive values
- When non-zero, DSWRFsfc has a very wide distribution (P1=0.03, P99=842 W/m2)

**Implication**: Standard Gaussian normalization is poorly suited for these bimodal distributions. The residual/full-field ratio < 1 means the residual normalization actually uses a *larger* scale than the full-field normalization, which is counterintuitive. Diurnal-cycle-aware normalization or conditional normalization (day/night) could improve learning.

---

## Finding 6: PRATEsfc has an extremely non-Gaussian distribution

- **18% exact zeros**, skewness **9.87**, kurtosis **198**
- The non-zero distribution spans 6 orders of magnitude (P1 of non-zero = 6.2e-9, P99 = 4.1e-4 kg/m2/s)
- Residual skewness = -0.37 but residual kurtosis = **166** (extreme outliers in precipitation changes)

This is expected physically (precipitation is inherently intermittent and heavy-tailed) but presents challenges for a model trained with MSE-like losses on Gaussian-normalized data.

---

## Finding 7: Temperature profile inversion at the tropopause

The vertical temperature profile is non-monotonic:

| Level | Pressure (hPa) | Global Mean T (K) | Tropics T (K) | Polar T (K) |
|-------|----------------|-------------------|---------------|-------------|
| 0 | 25.6 | 221.6 | 221.9 | 217.8 |
| 1 | 97.7 | **209.3** | **199.0** | 210.4 |
| 2 | 199.1 | 217.7 | 218.5 | 212.4 |
| 3 | 330.5 | 232.2 | 244.6 | 217.3 |
| 7 | 964.4 | 277.1 | 294.2 | 252.0 |

Level 1 (tropopause) is the coldest point in the profile. Level 0 (stratosphere) is 12K warmer globally, and 23K warmer in the tropics, due to ozone heating. This means:
- Temperature increases with height between levels 0 and 1 (opposite to the rest of the column)
- The physics governing levels 0-1 (stratospheric dynamics, ozone chemistry) is fundamentally different from levels 2-7 (tropospheric convection, radiation)
- The model must learn this transition without any explicit indication of which regime each level belongs to

---

## Finding 8: Signal is overwhelmingly spatial, not temporal

For all variables, the temporal variance of the global-mean 6h residual is less than 1% of the spatial variance of residuals:

| Variable | Temporal/Spatial Variance Ratio |
|----------|-------------------------------|
| air_temperature_0 | 0.0048 |
| specific_total_water_0 | 0.0016 |
| eastward_wind_0 | 0.0049 |
| northward_wind_3 | 0.0001 |

This means the global-mean state is nearly persistent from timestep to timestep. Almost all of what the model learns is about *where* changes happen spatially, not how much the globally-averaged state changes. The spatially-varying part of the prediction dominates the loss.

---

## Finding 9: Long-term trends in training data

Over the 1979-2022 training period:

| Variable | Trend per decade | Interpretation |
|----------|-----------------|---------------|
| air_temperature_0 | -0.47 K | Stratospheric cooling |
| air_temperature_7 | +0.21 K | Surface-layer warming |
| TMP2m | +0.23 K | Surface warming |
| specific_total_water_0 | -0.61%/decade | Stratospheric drying |
| specific_total_water_7 | +0.37%/decade | Lower troposphere moistening |
| eastward_wind_0 | +3.6%/decade | Strengthening stratospheric westerlies |
| USWRFsfc | -0.64%/decade | Changing surface albedo/clouds |

The normalization statistics are computed from 1990-2019, but training data extends back to 1979 and forward to 2021. The non-stationarity means early and late training samples have systematically different distributions from the normalization baseline. The stratospheric cooling trend (-0.47 K/decade over 43 years = ~2K shift) is comparable to the residual std (0.54 K), meaning the trend is a significant source of variance for air_temperature_0.
