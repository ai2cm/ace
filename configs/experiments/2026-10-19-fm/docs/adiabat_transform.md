# Adiabat transform

Change of variables applied before the usual (grouped) mean/std normalization.
Goal: inputs/outputs that shift less between climates and datasets
(piControl → 1pct/abrupt-CO2, ERA5 ↔ C96/CM4/X-SHiELD).

Level indexing: $k=0$ is the top (stratosphere), $k=7$ is the lowest level.

## 1. Reference parcel (per column, per timestep)

Inputs: $T_7$, $q_7$ (`specific_total_water_7`), $p_s$ (`PRESsfc`).
Level pressures: $p_k = a_k + b_k\,p_s$.

Saturation humidity (Bolton $e_s$, hPa):

$$
e_s(T) = 6.112\,\exp\!\left(\frac{17.67\,(T-273.15)}{T-29.65}\right),
\qquad
q_\text{sat}(T,p) = \frac{\varepsilon\,e_s(T)}{p-(1-\varepsilon)\,e_s(T)}
$$

Lift a parcel from $(T_7, q_7, p_7)$ upward:

1. **Dry ascent** (humidity fixed at $q_7$, below saturation):
   $$T_p(p) = T_7\left(\frac{p}{p_7}\right)^{R_d/c_p}$$
   until $q_\text{sat}(T_p, p) = q_7$ (saturation level). Skip if $q_7 \ge q_\text{sat}(T_7,p_7)$.
2. **Moist ascent** (saturated, excess vapor condenses and releases heat):
   $$
   \frac{dT_p}{dp} = \frac{1}{p}\,
   \frac{R_d T_p + L_v q_s}{c_p + \dfrac{\varepsilon L_v^2 q_s}{R_d T_p^2}},
   \qquad q_s = q_\text{sat}(T_p,p)
   $$
3. Record $T_p(p_k)$ for $k=1,\dots,6$.

## 2. Forward transform

| Variable | Transformed | Levels |
|---|---|---|
| Temperature | $D_k = T_k - T_p(p_k)$ | $k=1..6$ |
| Humidity | $H_k = q_k / q_\text{sat}(T_k, p_k)$ | $k=0..6$ |
| Everything else | unchanged | $T_0$, $T_7$, $q_7$, winds, surface, forcings |

$T_7, q_7, p_s$ stay raw because they define the parcel.
$H_k$ can exceed 1 in clouds ($q$ includes condensate); that's fine.

Then normalize as today, per group $g$: $\hat D_k = (D_k - \mu_{g,k})/\sigma_{g,k}$, same for $H_k$.

## 3. Inverse (after network output)

1. Rebuild $T_p(p_k)$ from predicted $T_7, q_7, p_s$.
2. $T_k = D_k + T_p(p_k)$.
3. $q_k = H_k\,q_\text{sat}(T_k, p_k)$ (needs step 2 first).

Exact inverse; loss computed on physical $T_k, q_k$ (keeps metrics comparable).

## 4. Implementation notes

- Differentiable torch op in `fme/core`, applied before normalizer / after denormalizer; config saved with stepper for checkpoint loading.
- Moist ascent: fixed RK substeps in $\ln p$ between levels; saturation level via Newton or Bolton LCL formula.
- Liquid-only latent heat to start (no ice).
- Code change goes in a PR off `main`, not the FM branch.

## 5. Plan

1. **Offline check (no training).** Per group, compute $D_k, H_k$. Compare gap
   $|\mu_a - \mu_b|/\sigma$ for raw vs transformed variables:
   piControl vs 1pct, ERA5 vs other groups. Tropics and extratropics separately.
   Stop if gaps don't clearly shrink.
2. **Decide levels.** Include/exclude $T_1$, $H_0$ based on step 1.
3. **One training arm** vs current baseline, only this change.
   Judge on abrupt-CO2 / 1pct generalization and cross-dataset skill, not 5-day RMSE.

## Risks

- Extratropics/poles not near moist adiabat → large, noisy $D_k$.
- Error in predicted $T_7, q_7$ propagates to all levels on inverse.
- Expected gain modest; mostly climate generalization, little weather skill.
