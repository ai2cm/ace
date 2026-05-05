# Bias Correction for CMIP6 Climate Model Data

This document reviews bias correction techniques commonly applied to climate
model output, with an emphasis on their relevance to ACE's CMIP6 emulator
and seasonal-to-decadal forecasting applications.

## Why bias correction matters

Climate models exhibit systematic biases relative to observations and
reanalysis. These stem from discretization, parameterization choices, and
numerical artifacts, and manifest as:

| Bias type | Description | Example |
|-----------|-------------|---------|
| Mean | Offset in climatological average | Model is 2 K cold in Arctic winter |
| Variance | Wrong amplitude of variability | Underestimated interannual temperature range |
| Distribution shape | Wrong skewness, tails, wet-day frequency | Too many drizzle days, too few extremes |
| Spatial pattern | Displaced large-scale features | Storm track shifted poleward |
| Temporal structure | Wrong persistence or seasonality | Wet/dry spell durations misrepresented |
| Inter-variable | Wrong correlations between variables | Temperature-humidity coupling incorrect |

For our project, bias correction is relevant in two distinct ways:

1. **Training-time normalization.** Each CMIP6 model has a different
   climatology. Cross-model normalization breaks when model climatologies
   differ too much — the `hus10` problem in `training.md` is a direct
   consequence. Per-model or targeted normalization is a lightweight form
   of bias correction that brings models into a shared input space.

2. **Inference-time debiasing.** If the emulator is trained on CMIP6 model
   output but initialized from ERA5, or if users want emulator output that
   matches observed statistics, a distribution mapping is needed at the
   input or output interface. This is the classical bias correction problem.

### Two framings of the problem

The literature distinguishes two motivations that map to our two use cases:

- **"Make model output look like observations"** — the traditional
  impact-modeling use case (ISIMIP, hydrology, agriculture). Map GCM output
  to an observational distribution so downstream models receive
  realistic-looking inputs.

- **"Transfer learning between distributions"** — more relevant for ML
  emulators. Standardize across models so a single network can learn shared
  dynamics despite model-specific biases. Closer to domain adaptation than
  classical bias correction.

Key references for why bias correction is needed and its limitations:
Maraun (2016, "Bias Correcting Climate Change Simulations — a Critical
Review," *Current Climate Change Reports*) and Ehret et al. (2012,
"Should we apply bias correction to global and regional climate model
data?" *HESS*).

---

## Common methods

### Simple scaling / delta method

Adjusts only the mean (and optionally variance).

- **Additive** (temperature): `x_corrected(t) = x_model(t) + [x̄_obs - x̄_model]`
- **Multiplicative** (precipitation): `x_corrected(t) = x_model(t) × [x̄_obs / x̄_model]`
- **Variance scaling** (Teutschbein & Seibert, 2012): after mean correction,
  scale deviations to match observed variance.

The delta method inherently preserves the climate change signal — it applies
the modeled *change* to the observed baseline. It cannot correct
distribution shape, tail behavior, or inter-variable relationships.

### Quantile mapping (QM) and variants

The workhorse family. Maps the model distribution to the observed
distribution quantile by quantile:

```
x_corrected = F_obs⁻¹(F_model(x))
```

where `F` denotes the CDF. Can be **empirical** (CDFs from sorted data,
simple but cannot extrapolate beyond the observed range) or **parametric**
(fit variable-specific distributions: normal for temperature, gamma for
precipitation, beta for bounded variables like relative humidity, Weibull
for wind speed).

**Detrended QM (DQM)** — Removes the long-term trend before QM, then
re-imposes it (Cannon et al., 2015; Boe et al., 2007). Preserves the
trend in the mean but not in higher moments.

**Quantile Delta Mapping (QDM)** — Preserves the model-projected change at
*every* quantile, not just the mean (Cannon et al., 2015). For a future
model value at quantile τ:

1. Compute the model-projected change at that quantile:
   `Δ(τ) = F_model_future⁻¹(τ) - F_model_hist⁻¹(τ)`
2. Apply to the bias-corrected historical:
   `x_corrected = F_obs⁻¹(τ) + Δ(τ)`

QDM is recommended by multiple comparative studies as the best default when
trend preservation across the full distribution matters. Mathematically
equivalent to Equidistant CDF Matching (Li et al., 2010).

**CDF-t** (Michelangeli et al., 2009) — Derives a transfer function T from
the calibration period such that `F_obs = T(F_model_hist)`, then applies it
to the future: `F_corrected_future = T(F_model_future)`.

All QM variants are **univariate** (applied independently per variable and
grid cell), which can break inter-variable relationships and spatial
coherence. They also assume stationarity of the bias (the calibration-period
transfer function holds in the future).

### ISIMIP3BASD: parametric trend-preserving QM

The ISIMIP3BASD framework (Lange, 2019) is the de facto standard for
bias-correcting CMIP6 data for impact studies. It assigns
variable-specific parametric distributions:

| Variable | Distribution | Bounds |
|----------|-------------|--------|
| `tas`, `tasmax`, `tasmin` | Normal | Unbounded |
| `pr` | Point mass at 0 + parametric tail | ≥ 0 |
| `rsds` | Beta | [0, TOA_max] |
| `hurs` | Beta | [0%, 100%] |
| `sfcWind` | Weibull | ≥ 0 |
| `ps`, `psl` | Normal | Unbounded |

The method:
1. Estimates long-term monthly means and trends for model and observations.
2. Removes the model trend (preserving it).
3. Applies parametric QM to the detrended day-to-day variability per
   calendar month with a 31-day running window.
4. Re-imposes the trend (additively for temperature, multiplicatively for
   precipitation).

Reference data is W5E5 v2.0 (WFDE5 over land merged with ERA5 over ocean).
The published ISIMIP3b product covers 11 surface variables at 0.5° for
major CMIP6 models — but **no pressure-level data**, so it cannot be used
directly for our pipeline.

### Multivariate methods

Univariate methods correct each variable independently, potentially
producing physically impossible states (e.g., supersaturation after
correcting T and q separately). Multivariate methods address the joint
distribution.

**MBCn** (Cannon, 2018) — Iteratively adjusts the dependence structure:
1. Apply QDM univariately to correct marginal distributions.
2. Multiply data by random orthogonal rotation matrices to decorrelate.
3. Apply QDM to rotated data.
4. Inverse-rotate and repeat until convergence.

Borrowed from the N-pdft image processing algorithm for color transfer
between images.

**R2D2** (Vrac, 2018; v2.0: Vrac & Thao, 2020) — Corrects marginals
univariately, then adjusts dependence by resampling ranks from
observations (Schaake Shuffle). Scales better than MBCn for
high-dimensional problems. v2.0 adds temporal dependence correction.

**dOTC** (Robin et al., 2019) — Uses optimal transport theory to map
multivariate distributions while accounting for non-stationarity.

Francois et al. (2020) compared MBCn, R2D2, and dOTC and found that
multivariate methods significantly improve inter-variable relationships,
but no single method dominates, and all risk degrading temporal structure
unless explicitly constrained.

### Machine learning approaches

**Deep learning for spatial bias correction.** Hess et al. (2023, "Deep
Learning for Bias-Correcting CMIP6-Class Earth System Models," *Earth's
Future*) used physically-constrained conditional GANs to correct
precipitation from GFDL-ESM4. Key advantage: corrects *spatial patterns*,
not just pointwise distributions. Standard QM corrects marginals equally
well but cannot fix unrealistic spatial patterns.

**State-dependent ML correction.** Zhang et al. (2024, *JAMES*) used
LSTMs for bias correction of E3SM, learning different corrections
depending on the atmospheric state rather than a static transfer function.

**Trend-preserving deep learning.** Qin et al. (2024, *Climate Dynamics*)
combined deep learning with explicit trend-preservation constraints,
addressing the concern that purely data-driven approaches cannot reliably
extrapolate to unseen future climates.

---

## Software packages

| Package | Language | Methods | Multivariate | Notes |
|---------|----------|---------|-------------|-------|
| **xclim/xsdba** | Python | EQM, DQM, QDM, Scaling, N-pdft, Extreme Value | Yes (N-pdft) | Most comprehensive Python library; Dask-parallel; part of xclim ecosystem |
| **python-cmethods** | Python | Linear Scaling, Variance Scaling, Delta, QM, DQM, QDM | No | Clean API, xarray/Dask, easy to integrate |
| **ibicus** | Python | Delta, Linear Scaling, QM, QDM, CDFt, EDCDFm, ISIMIP3BASD | No | Strong evaluation framework alongside methods (Spuler et al., 2024) |
| **ISIMIP3BASD** | Python | Parametric trend-preserving QM | No | De facto CMIP6 standard for impact studies |
| **MBC** | R | QDM, MBCp, MBCn, MBCr, R2D2 | Yes | Reference implementation by A. Cannon |
| **SBCK** | Python/R | Optimal transport (dOTC), QM variants, multivariate | Yes | Includes optimal-transport methods |
| **CDO** | C (CLI) | Basic arithmetic scaling | No | Primarily a regridding/preprocessing tool |

For our Python-based pipeline, **xclim/xsdba** is the most natural choice
for prototyping: comprehensive method coverage, xarray-native, Dask
parallelism, multivariate support, and active maintenance. **ibicus** is
valuable for systematic evaluation of which method works best for our
variables.

---

## Bias correction for seasonal-to-decadal prediction

Seasonal-to-decadal (S2D) prediction has a different bias correction
problem than long-range climate projections. Initialized predictions drift
away from the observed state toward the model's own climatology, and this
drift is *lead-time dependent*.

**Lead-time-dependent mean correction.** Estimate the mean bias as a
function of lead time from hindcasts, typically using polynomial fits in
lead time (Manzanas et al., 2020; Pasternack et al., 2021).

**Anomaly initialization.** Initialize the model on its own attractor (add
observed anomalies to model climatology) to reduce initial drift. Avoids
the drift correction problem but may sacrifice predictive skill.

**Ensemble calibration.** S2D systems use ensembles. Bias correction must
handle ensemble mean bias *and* spread (which may be over- or
under-dispersive), both as functions of lead time. The standard approach
is "calibrated anomalies": anomalies relative to the lead-time-dependent
model climatology, added to the observed climatology.

For an ML emulator intended for S2D prediction, the key challenge is
**distribution mismatch at initialization**: if trained on CMIP6 model
output but initialized from ERA5, the emulator receives inputs from a
distribution it has not seen. This is analogous to full-field
initialization drift — the emulator will drift toward the training
distribution's climatology. The stationarity assumption is more defensible
at S2D timescales (modest climate change over the forecast horizon)
compared to end-of-century projections.

---

## Important considerations and debates

### Stationarity of bias

All bias correction methods assume that the model-observation relationship
from the calibration period holds in the future. Ehret et al. (2012) and
Maraun (2016) articulate the core problem: "The bias itself can be
entangled with a climate signal, with no real possibility of discriminating
between the two." If the bias depends on the mean state, a transfer
function learned in a cool climate may not apply in a warmer one. For S2D
timescales this is less of a concern than for end-of-century projections.

### Preserving the climate change signal

Standard QM corrupts the projected signal by mapping future values through
the historical transfer function. Linear scaling perfectly preserves mean
trends but ignores distributional shifts. QDM and ISIMIP3BASD attempt to
preserve changes at every quantile. Even so, Spuler et al. (2024) found
that in practice the magnitude of the trend can be modified. This matters
when bias-correcting training data for an emulator that should learn the
correct forced response.

### Univariate vs multivariate

At our ~4° resolution with ~50+ variables (8 pressure levels × 4 core
variables + surface + derived), the dimensionality is high. MBCn's
iterative rotations scale poorly; R2D2 scales better but makes strong
assumptions about the dependence structure. However, if the ML emulator
itself learns inter-variable relationships through its architecture, then
univariate pre-correction of inputs may be adequate, with the network
providing multivariate consistency.

### Physical consistency after correction

Bias correction is a statistical fix to a physical problem. After
correction, energy budgets may not close, mass conservation may be
violated, and geostrophic balance may be broken. For ML training this is
particularly concerning: if bias-corrected data breaks the thermodynamic
consistency between T, q, z, and winds, the emulator learns a
non-physical state space. This argues for either (a) minimal correction
(per-model normalization) or (b) correction methods that explicitly
preserve physical constraints.

### Cross-validation is misleading

Maraun et al. (2018, *HESS*) showed that cross-validating bias correction
against observations is fundamentally misleading: residual bias depends
primarily on the difference between simulated and observed internal
variability realizations, not on the correction method's quality. For S2D
prediction, hindcast skill assessment is the best available validation.

---

## How we might apply bias correction

### Use case 1: improving multi-model training (training-time)

**Problem.** Cross-model normalization breaks when models have very
different climatologies. The `hus10` issue in `training.md` — where models
whose stratospheric humidity is far from the cross-model mean arrive as
outlier inputs — is a distribution mismatch problem.

**Options, from lightest to heaviest:**

1. **Per-model z-scoring for problematic variables.** For variables with
   large inter-model spread (stratospheric humidity, and potentially
   others), normalize each model to its own mean and standard deviation
   rather than the multi-model mean. The model embedding provides the
   information needed to reconstruct absolute values. This is the
   simplest approach and already partially explored.

2. **Per-model quantile mapping to a reference distribution.** For each
   variable, map each model's distribution to a common reference (e.g.,
   the multi-model median distribution or ERA5) before training. More
   aggressive than z-scoring — corrects distribution shape, not just
   mean and variance. Risk: may mask genuine physical differences
   between models that the label embedding should capture.

3. **Hierarchical normalization.** Per-model statistics regularized toward
   the multi-model mean. Prevents extreme outliers (IITM-ESM) from
   dominating the loss while preserving model-specific structure. A
   middle ground between cross-model and fully per-model normalization.

4. **Add ERA5 as a training "model."** Include ERA5 data with its own
   label in the training set. The network learns the ERA5 distribution
   alongside CMIP6 models. At inference time, use the ERA5 label. No
   explicit bias correction needed; the distribution gap is handled by
   the embedding. This naturally enables transfer to ERA5 initial
   conditions.

**Recommendation.** Start with per-model normalization (option 1) for the
immediate `hus10` problem. Option 4 (ERA5 as a training model) is the
most promising longer-term approach — it sidesteps the bias correction
problem entirely and is architecturally clean.

### Use case 2: debiased emulator output for users (inference-time)

**Problem.** Users running the emulator for seasonal-to-decadal forecasts
want output that looks like observations (ERA5/reanalysis), not like
a biased CMIP6 model.

**Options:**

1. **Post-hoc QDM on emulator output.** Train the emulator on raw CMIP6
   data, then apply Quantile Delta Mapping as a post-processing step
   to map the output distribution to ERA5. This preserves the emulator's
   learned dynamics while adjusting the output statistics. QDM is
   preferred over basic QM because it preserves the emulator's projected
   changes at each quantile.

2. **Train with ERA5 label, emit debiased output directly.** If ERA5 is
   included as a training "model" (use case 1, option 4), the emulator
   with the ERA5 label naturally produces output in the ERA5 distribution.
   No post-processing needed.

3. **Learn a debiasing head.** Add a lightweight learned mapping (e.g., a
   per-variable affine transform conditioned on the label) that maps from
   model space to observation space. Trained on paired model-ERA5 data.

**Recommendation.** Option 2 is the cleanest if training with ERA5 works.
Otherwise, option 1 (post-hoc QDM) is well-understood and can be applied
per-variable with standard tools (xclim/xsdba).

### Use case 3: ERA5 initialization of a CMIP6-trained emulator

**Problem.** If the emulator is trained on CMIP6 data and initialized
from ERA5, the input distribution mismatch causes the emulator to drift
toward the training distribution's climatology.

**Options:**

1. **Inverse bias correction at initialization.** Transform ERA5 initial
   conditions into the nearest CMIP6 model's distribution space, run the
   emulator, then transform back. Requires choosing a reference model
   and applying per-variable QM.

2. **Fine-tune on ERA5.** Pre-train on CMIP6, then fine-tune the emulator
   (or a subset of parameters) on ERA5 data. Standard transfer learning.

3. **ERA5 as a training model (same as use case 1, option 4).** The
   emulator already knows the ERA5 distribution. Initialize with ERA5
   data and the ERA5 label. No distribution mapping needed.

4. **Lead-time-dependent drift correction.** Run the emulator from ERA5
   initial conditions, then correct the output drift as a function of
   lead time, calibrated from "hindcasts" where the emulator is run from
   historical ERA5 states and compared to subsequent ERA5 states.

**Recommendation.** Again, option 3 (ERA5 as a training model) is the most
architecturally clean solution. If that is not feasible, fine-tuning
(option 2) is the most proven ML approach. Lead-time-dependent drift
correction (option 4) is the standard approach in operational S2D
prediction and could complement either option.

---

## Summary of recommendations

1. **Do not bias-correct training data to observations.** This would mask
   inter-model variability that the label embedding is designed to capture,
   and risks breaking physical consistency.

2. **Use per-model normalization** (already partially in place) to handle
   the inter-model spread problem. This is a lightweight, transparent form
   of distribution standardization.

3. **Include ERA5 as a training "model"** to naturally bridge the
   model-reanalysis distribution gap. This is the single highest-leverage
   step for enabling both debiased output and ERA5-initialized forecasts.

4. **For inference-time debiasing**, apply QDM post-processing or rely on
   the ERA5 training label. Either approach preserves the emulator's
   learned dynamics.

5. **For S2D forecasting**, consider lead-time-dependent drift correction
   calibrated from hindcasts, which is standard practice in operational
   prediction systems and complementary to the above.

---

## Key references

### Foundational / review

- Maraun, D. (2016). "Bias Correcting Climate Change Simulations — a Critical Review." *Current Climate Change Reports*, 2, 211–220.
- Ehret, U. et al. (2012). "Should we apply bias correction to global and regional climate model data?" *HESS*, 16, 3391–3404.
- Maraun, D. & Widmann, M. (2018). *Statistical Downscaling and Bias Correction for Climate Research.* Cambridge University Press.
- Teutschbein, C. & Seibert, J. (2012). "Bias correction of regional climate model simulations for hydrological climate-change impact studies." *J. Hydrology*, 456–457, 12–29.

### Method papers

- Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). "Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?" *J. Climate*, 28(17), 6938–6959. (QDM, DQM)
- Cannon, A. J. (2018). "Multivariate quantile mapping bias correction: an N-dimensional probability density function transform." *Climate Dynamics*, 50, 31–49. (MBCn)
- Lange, S. (2019). "Trend-preserving bias adjustment and statistical downscaling with ISIMIP3BASD (v1.0)." *Geosci. Model Dev.*, 12, 3055–3070.
- Vrac, M. (2018). "Multivariate bias adjustment of high-dimensional climate simulations: the R2D2 bias correction." *HESS*, 22, 3175–3196.
- Vrac, M. & Thao, S. (2020). "R2D2 v2.0: accounting for temporal dependences." *GMD*, 13, 5367–5387.
- Li, H., Sheffield, J., & Wood, E. F. (2010). "Bias correction of monthly precipitation and temperature fields using equidistant quantile matching." *J. Geophys. Res.*, 115. (EDCDFm)
- Michelangeli, P. A., Vrac, M., & Loukos, H. (2009). "Probabilistic downscaling approaches: Application to wind CDFs." *Geophys. Res. Lett.*, 36. (CDF-t)
- Robin, Y. et al. (2019). "Multivariate stochastic bias corrections with optimal transport." *HESS*, 23, 773–786. (dOTC)
- Piani, C. et al. (2010). "Statistical bias correction for daily precipitation in regional climate models over Europe." *Theor. Appl. Climatol.*, 99, 187–192.

### Evaluations

- Francois, B. et al. (2020). "Multivariate bias corrections of climate simulations: which benefits for which losses?" *Earth System Dynamics*, 11, 537–562.
- Spuler, F. R. et al. (2024). "ibicus: a new open-source Python package for statistical bias adjustment and evaluation (v1.0.1)." *GMD*, 17, 1249–1269.
- Maraun, D. et al. (2018). "Cross-validation of bias-corrected climate simulations is misleading." *HESS*, 22, 4867–4873.

### ML approaches

- Hess, P. et al. (2023). "Deep Learning for Bias-Correcting CMIP6-Class Earth System Models." *Earth's Future*, 11.
- Zhang, C. et al. (2024). "A Machine Learning Bias Correction on Large-Scale Environment of High-Impact Weather Systems in E3SM." *JAMES*.
- Qin, Y. et al. (2024). "Multivariate bias correction and downscaling of climate models with trend-preserving deep learning." *Climate Dynamics*.
- Nguyen, T. et al. (2023). "ClimaX: A foundation model for weather and climate." *ICML*.

### Seasonal/decadal prediction

- Manzanas, R. et al. (2020). "Assessment of Model Drifts in Seasonal Forecasting." *JAMES*.
- Pasternack, A. et al. (2021). "Recalibrating decadal climate predictions — what is an adequate model for the drift?" *GMD*, 14, 4335–4355.
- Kharin, V. et al. (2012). "Evaluation of forecast strategies for seasonal and decadal forecasts in presence of systematic model errors." *Climate Dynamics*.
