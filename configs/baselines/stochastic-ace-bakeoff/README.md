# Stochastic-ACE baseline bake-off

Phase 0 of the stochastic-ACE paper. Five training runs on the 1°/daily
ERA5-only dataset that vary only the loss weighting, the total-energy
corrector, and spectral whitening of the energy score, to pick the
stochastic-ACE baseline recipe. Runs are launched by Jeremy via
`run-train.sh` — this directory only defines the configs.

## Arms

All five arms share one base config (arm 1). Arms differ only in the
knobs below.

| config | crps_weight | energy_score_weight | total-energy corrector | energy-score whitening |
|---|---|---|---|---|
| `arm1-90-10-ec.yaml` (base) | 0.9 | 0.1 | `constant_temperature` | none |
| `arm2-90-10-noec.yaml` | 0.9 | 0.1 | off | none |
| `arm3-50-50-ec.yaml` | 0.5 | 0.5 | `constant_temperature` | none |
| `arm4-90-10-ec-whiten-g0.5.yaml` | 0.9 | 0.1 | `constant_temperature` | `per_sample`, γ=0.5 |
| `arm5-50-50-ec-whiten-g0.5.yaml` | 0.5 | 0.5 | `constant_temperature` | `per_sample`, γ=0.5 |

`crps_weight`/`energy_score_weight` and `energy_score_whitening` live in
`stepper_training.loss.kwargs`; the corrector toggle is
`stepper.step.config.corrector.total_energy_budget_correction`
(`constant_unaccounted_heating` defaults to 0.0). Every arm uses
`seed: 0`.

## Base recipe

Recovered from Troy's run `nzccs8zd`
(https://wandb.ai/ai2cm/ace/runs/nzccs8zd): NoiseConditionedSFNO
(embed_dim 512, 8 layers, spectral_ratio 0.125, isotropic noise,
`clip_latent_global_means` off), residual prediction, EnsembleLoss
(n_ensemble 2, n_forward_steps 1), shared global-mean removal, dry-air +
moisture-budget correction, EMA 0.999, FusedAdam lr 1e-4, batch_size 8.
40 inputs / 51 outputs: the four near-surface fields (TMP2m, Q2m,
UGRD10m, VGRD10m) are output-only diagnostics, not inputs, and the model
predicts `total_frozen_precipitation_rate`. Trained 80 epochs on
`/climate-default/2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr` with
an `h500: 5` per-channel loss weight. Data is 06Z daily, so every
inference initial condition is at `T06:00:00`.

In-training inference monitors 10-year stability/day-5 SSR/CRPS
(`10year`, `10year_insample`, `weather_2024`, `weather_1994`). The
46-year rollout is not run per arm; it and the full selection metrics
(10-yr bias, day-5 SSR, climate-skill overview, spectral power) are a
dedicated offline eval on the selected winner after the runs finish.

## Train/validation split (stitch-aware ACE2 split)

Train window is the ACE2 window (1940–1995, 2011–2019, 2021–2025), but
split at the 11 ERA5 production-stream boundaries so that no residual
training sample straddles a stream stitch (a stitch produces a spurious
tendency). This yields 14 `subset` segments in `train_loader`. Validation
is 1996–1997 (a held-out gap year pair between the two train blocks).
