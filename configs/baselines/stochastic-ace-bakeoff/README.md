# Stochastic-ACE baseline bake-off

Phase 0 of the stochastic-ACE paper. Eight training runs on the 1°/daily
ERA5-only dataset that vary only the loss weighting (CRPS, energy score,
and spectral-power CRPS), the total-energy corrector, and spectral
whitening, to pick the stochastic-ACE baseline recipe. Runs are launched
by Jeremy via `run-train.sh` — this directory only defines the configs.

## Arms

All eight arms share one base config (arm 1). Arms differ only in the
knobs below. The weight columns are `crps` (`crps_weight`), `es`
(`energy_score_weight`), and `sp` (`spectral_power_crps_weight`).

| config | crps | es | sp | total-energy corrector | whitening |
|---|---|---|---|---|---|
| `arm1-90-10-ec.yaml` (base) | 0.9 | 0.1 | 0 | `constant_temperature` | none |
| `arm2-90-10-noec.yaml` | 0.9 | 0.1 | 0 | off | none |
| `arm3-50-50-ec.yaml` | 0.5 | 0.5 | 0 | `constant_temperature` | none |
| `arm4-90-10-ec-whiten-g0.5.yaml` | 0.9 | 0.1 | 0 | `constant_temperature` | `per_sample`, γ=0.5 |
| `arm5-50-50-ec-whiten-g0.5.yaml` | 0.5 | 0.5 | 0 | `constant_temperature` | `per_sample`, γ=0.5 |
| `arm6-80-10-sp10-ec.yaml` | 0.8 | 0.1 | 0.1 | `constant_temperature` | none |
| `arm7-90-0-sp10-ec.yaml` | 0.9 | 0.0 | 0.1 | `constant_temperature` | none |
| `arm8-80-10-sp10-ec-whiten-g0.5.yaml` | 0.8 | 0.1 | 0.1 | `constant_temperature` | `per_sample`, γ=0.5 |

`crps_weight`/`energy_score_weight`/`spectral_power_crps_weight` and
`energy_score_whitening` live in `stepper_training.loss.kwargs`; the
corrector toggle is
`stepper.step.config.corrector.total_energy_budget_correction`
(`constant_unaccounted_heating` defaults to 0.0). Every arm uses
`seed: 0`. Arm 8 shares its γ=0.5 whitening operator between the energy
score and the spectral-power CRPS term (the same reweight applies to
both).

## Spectral-power CRPS term

Arms 6–8 add a spectral-power-CRPS loss term
(`SpectralPowerCRPSLoss`, `spectral_power_crps_weight`), an unmerged
`fme/core/loss.py` feature this experiment branch integrates. It scores
the (almost-)fair CRPS of the per-degree log spectral power, so it is
phase-free (per-sample spatial phase noise does not enter) and
scale-equitable across the red spectrum, giving a high-SNR amplitude
gradient at degrees where per-mode energy-score gradients are
noise-starved. It was validated at small scale in the 2026-07-06
small-scale-calibration report (arm 2 there = `5k3fmlif`). When
`energy_score_whitening` is enabled (arm 8), the per-degree power CRPS is
reweighted with the same whitening operator the energy score uses.

## Base recipe

Recovered from Troy's run `nzccs8zd`
(https://wandb.ai/ai2cm/ace/runs/nzccs8zd): NoiseConditionedSFNO
(embed_dim 512, 8 layers, spectral_ratio 0.125, isotropic noise,
`clip_latent_global_means` off), EnsembleLoss
(n_ensemble 2, n_forward_steps 1), dry-air +
moisture-budget correction, EMA 0.999, FusedAdam lr 1e-4, batch_size 8.
40 inputs / 51 outputs: the four near-surface fields (TMP2m, Q2m,
UGRD10m, VGRD10m) are output-only diagnostics, not inputs, and the model
predicts `total_frozen_precipitation_rate`. Trained 80 epochs on
`/climate-default/2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr` with
an `h500: 5` per-channel loss weight. Data is 06Z daily, so every
inference initial condition is at `T06:00:00`.

## Checkpoint selection and in-training inference

Checkpoint selection is driven by a single weight-1.0 inference loop,
`ace2_5yr_1996`, matching the ACE2-paper's selection inference: eight
out-of-sample initial conditions spread through 1996 (inside the held-out
1996–2010 gap), each rolled out 5 years deterministically
(`n_forward_steps: 1826` at daily cadence, `n_ensemble_per_ic: 1`, ICs at
06Z). Its selection metric is the time-mean of the inference period scored
against the rollout's own target (`time_mean_norm/rmse/channel_mean`); no
external time-mean reference is used. The full 5-year rollout stays inside
the held-out gap, so selection is entirely out-of-sample.

The `10year`, `10year_insample`, `weather_2024`, and `weather_1994` loops
are retained as weight-0 diagnostics (10-year stability, day-5 SSR/CRPS)
and no longer influence selection. The 46-year rollout is not run per arm;
it and the full selection metrics (10-yr bias, day-5 SSR, climate-skill
overview, spectral power) are a dedicated offline eval on the selected
winner after the runs finish.

## Train/validation split (stitch-aware ACE2 split)

Train window matches the ACE2-paper data span (1940–1995, 2011–2019,
2021–2022): the final segment ends 2022-12-31 (the ACE2 data ended there),
not 2025. It is split at the ERA5 production-stream boundaries so that no
residual training sample straddles a stream stitch (a stitch produces a
spurious tendency); the daily-dataset seam report confirmed no stitch
boundary falls in 2021–2025, so capping the final segment at 2022 needs no
new split. This yields 14 `subset` segments in `train_loader`. Validation
is 1996–1997 (a held-out gap year pair between the two train blocks),
unchanged.

## Correction (2026-08-07): residual prediction and global-mean removal

The "Base recipe" section above previously credited the donor run
[nzccs8zd](https://wandb.ai/ai2cm/ace/runs/nzccs8zd) with "residual
prediction" and "shared global-mean removal". **Both were wrong.** That run's
config sets neither key — it is the main ERA5 baseline recipe with exactly two
architecture knobs changed, `filter_num_groups: 16` and
`spectral_ratio: 0.125`, which is what its name records
(`ace2s-era5-daily-spectral-groups-16-ratio-0.125-1-step-pre-training-rs0`).

These eight arms carry `residual_prediction: true` and a shared
`global_mean_removal` block because they adopted the 4°/daily **v2**
architecture block wholesale rather than building base + those two knobs;
their `stepper.step.config.builder.config` is byte-for-byte v2's, minus
`clip_latent_global_means`. The 6h subset dropped `global_mean_removal` on
2026-07-29 and `residual_prediction` on 2026-08-07; these daily arms are
finished and are left as they ran.

## Residual ablation arm (2026-08-07)

`arm1-90-10-ec-nores.yaml` is `arm1-90-10-ec.yaml` with
`residual_prediction: true → false` and **nothing else** — one line, verified
by diff. It is a strict one-knob A/B against the completed arm 1
([qhv9zf95](https://wandb.ai/ai2cm/ace/runs/qhv9zf95)) testing whether this
wave's small-scale precipitation power deficit
([reports#51](https://github.com/ai2cm/reports/pull/51)) is caused by residual
prediction.

`global_mean_removal` and the `8`/`4` loader knobs are deliberately **kept**,
matching qhv9zf95, so the comparison stays one-knob — this arm is a diagnostic
against the daily wave, not a member of the corrected 6h recipe. It runs on
8 GPUs where qhv9zf95 ran on 4; `batch_size: 8` is the global batch
(`dist.local_batch_size` divides by rank count), so the effective batch and
gradient are unchanged.

The eight arms above are complete — launch this one alone:
`./run-train.sh nores`.
