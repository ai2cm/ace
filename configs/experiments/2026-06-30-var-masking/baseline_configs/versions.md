# Baseline config versions

Each version is a full baseline config that the generators source from. Only
the co2-input change (v1 -> v2) affects config generation; everything else is
copied through unchanged into the swept configs.

## Comparison table

Common to all five versions (v1–v5): `seed: 0`, `max_epochs: 150`,
`ema.decay: 0.999`, `validate_using_ema: true`, `lr: 0.0001`,
`train_loader.batch_size: 8`, `validation_loader.batch_size: 32`,
`num_data_workers: 4` everywhere, `stepper_training.n_forward_steps: 1`,
`n_ensemble: 2`, `EnsembleLoss` (crps 0.9 / energy 0.1), `NoiseConditionedSFNO`
with `embed_dim: 512` / `num_layers: 8` / `dhconv`, 8 vertical layers, wandb
project `VarMasking8`, and validation-loader subsets 1994 + 2014.

| Aspect | v1 | v2 | v3 | v4 | v5 |
| --- | --- | --- | --- | --- | --- |
| File `ace2-var-mask-nc-sfno-era5-*.yaml` | `-v1` | `-v2` | `-v3` | `-v4` | `-v5` |
| `global_mean_co2` a network input (`in_names` + `next_step_forcing_names`) | yes | no | no | no | no |
| Generator sweep axes | var-masking × co2 × gmr (20) | var-masking × gmr (10) | var-masking × gmr (10) | var-masking × gmr (10) | var-masking × gmr (10) |
| Band-limited SFNO (`filter_num_groups: 16`, `spectral_ratio: 0.125`) | yes | yes | **no** | yes | **no** |
| Training window | 1979–2013, 1994 held out | 1979–2013, 1994 held out | 1979–2008 contiguous | 1979–2008 contiguous | 1979–2008 contiguous |
| Inference entry with `weight: 1.0` (drives validation score) | `10year_insample` | `10year_insample` | `aimip_checkpoint` | `aimip_checkpoint` | `aimip_checkpoint` |
| `aimip_checkpoint` entry (8 ICs in 2009, denorm+norm means on) | absent | absent | present | present | present |
| `long_46year_constant_co2` entry (`persistence_names: [global_mean_co2]`) | present | present | present | present | **removed** |
| Dataset | `2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr` | same | same | same | `2026-03-19-era5-1deg-8layer-1940-2025.zarr` |
| Grid / cadence | 4°, 1 step/day | 4°, 1 step/day | 4°, 1 step/day | 4°, 1 step/day | 1°, 4 steps/day |
| `n_forward_steps`: 10year / aimip / weather / long_46year | 3652 / – / 5 / 16794 | 3652 / – / 5 / 16794 | 3652 / 1825 / 5 / 16794 | 3652 / 1825 / 5 / 16794 | 14608 / 7300 / 20 / 67176 |
| `forward_steps_in_memory` (10year / aimip / long_46year) | 73 | 73 | 73 | 73 | 40 |
| Aggregator `step_means` + `ensembles` lead time | `step: 5` | `step: 5` | `step: 5` | `step: 5` | `step: 20` |
| Submission footprint (`submit_seed_jobs.py`) | `N_GPUS=2`, 100GiB | same | same | same | `N_GPUS=8`, 400GiB |

## v1 — `ace2-var-mask-nc-sfno-era5-v1.yaml`

Original var-masking baseline. `global_mean_co2` is a network input, listed in
both `stepper.step.config.next_step_forcing_names` and `...in_names`.

## v2 — `ace2-var-mask-nc-sfno-era5-v2.yaml`

Same as v1, except `global_mean_co2` is removed from `next_step_forcing_names`
and `in_names` (no longer a network input). The `long_46year_constant_co2`
inference entry still keeps `persistence_names: [global_mean_co2]`.

Because co2 is no longer an input, the generators drop the co2-masking axis for
v2+, leaving var-masking × global-mean-removal (`gmron`/`gmroff`); the
global-mean-removal axis is swept for v1 as well.

## v3 — `ace2-var-mask-nc-sfno-era5-v3.yaml`

Co2 handled exactly as in v2 (not an input; generation is identical). Differs
from v2 only in the baseline model/data setup, copied through unchanged:

- **Model**: `builder` drops `filter_num_groups` and `spectral_ratio`.
- **Training data**: window is 1979–2008 contiguous (v2 was 1979–2013 with 1994
  held out); i.e. shorter tail, 1994 no longer skipped.
- **Validation scoring**: the validation weight moves off `10year_insample`
  onto a new `aimip_checkpoint` inference entry (8 ICs in 2009, 1825 steps,
  denorm/norm means enabled).

## v4 — `ace2-var-mask-nc-sfno-era5-v4.yaml`

Identical to v3 (AIMIP protocol: no co2 input, 1979–2008 training window,
`aimip_checkpoint` validation), **except the band-limited SFNO backbone is
restored**: `builder` re-adds
`filter_num_groups: 16` and `spectral_ratio: 0.125` (which v3 had dropped).

## v5 — `ace2-var-mask-nc-sfno-era5-v5.yaml`

Baseline for the paper. Data paths point at the 1-degree, native 6-hourly ERA5 dataset
(`2026-03-19-era5-1deg-8layer-1940-2025.zarr`, the newest 6-hourly 1-degree
drop available) instead of the 4-degree daily-averaged one, with matching
1-degree normalization stats. Same 8 vertical layers, so variable lists are
unchanged.

Cadence goes from 1 step/day to 4 steps/day, so every step-counted field
that encodes a real-world span (`n_forward_steps`, aggregator
`step_means`/`ensembles` lead time) is multiplied by 4 to keep the same
calendar coverage — the `day_5`/`day_5_norm`/`day_5_ensemble` names stay
accurate, since 20 steps × 6 h = 5 days. `forward_steps_in_memory` drops
73 -> 40 on the long runs (the weather entries track `n_forward_steps` at 20).
`stepper_training.n_forward_steps` stays at 1, so v5 trains a native
6-hourly-timestep model rather than a daily one.

Two further differences from v4, beyond the resolution change:

- The `long_46year_constant_co2` inference entry (and its
  `persistence_names: [global_mean_co2]`) is dropped; only `long_46year`
  remains, so no config in v5 references `global_mean_co2` at all.
- The `builder` block does **not** carry `filter_num_groups: 16` /
  `spectral_ratio: 0.125`, so the paper baseline is un-band-limited like v3,
  not band-limited like v4.

Batch size (8 train / 32 val), worker count (4), and learning rate (0.0001) are
unchanged from v1–v4 and have not been re-tuned for the larger grid. Model
hyperparameters are otherwise copied through from v4. The submission overrides
for the 1-degree footprint (`N_GPUS=8`, `--shared-memory 400GiB`) live in
`submit_seed_jobs.py`, not the config.
