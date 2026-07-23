# Baseline config versions

Each version is a full baseline config that the generators source from. Only
the co2-input change (v1 -> v2) affects config generation; everything else is
copied through unchanged into the swept configs.

## v1 — `ace2-var-mask-nc-sfno-era5-v1.yaml`

Original var-masking baseline. `global_mean_co2` is a network input, listed in
both `stepper.step.config.next_step_forcing_names` and `...in_names`.

## v2 — `ace2-var-mask-nc-sfno-era5-v2.yaml`

Same as v1, except `global_mean_co2` is removed from `next_step_forcing_names`
and `in_names` (no longer a network input). The `long_46year_constant_co2`
inference entry still keeps `persistence_names: [global_mean_co2]`.

Because co2 is no longer an input, the generators drop the co2-masking axis for
v2+ and instead sweep global-mean-removal (`gmron`/`gmroff`).

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

Recommended baseline for the paper. Identical to v3 (AIMIP protocol: no co2
input, 1979–2008 training window, `aimip_checkpoint` validation), **except the
band-limited SFNO backbone is restored**: `builder` re-adds
`filter_num_groups: 16` and `spectral_ratio: 0.125` (which v3 had dropped).

## v5 — `ace2-var-mask-nc-sfno-era5-v5.yaml`

Identical to v4, except data paths point at the 1-degree, native 6-hourly
ERA5 dataset (`2026-03-19-era5-1deg-8layer-1940-2025.zarr`, the newest
6-hourly 1-degree drop available) instead of the 4-degree daily-averaged
one. Same 8 vertical layers, so variable lists are unchanged.

Cadence goes from 1 step/day to 4 steps/day, so every step-counted field
that encodes a real-world span (`n_forward_steps`, `forward_steps_in_memory`,
aggregator `step_means`/`ensembles` lead time) is multiplied by 4 to keep the
same calendar coverage. `stepper_training.n_forward_steps` stays at 1, so v5
trains a native 6-hourly-timestep model rather than a daily one.

Batch size (16 train / 128 val), worker count (16), and learning rate
(0.0001) match the 1-degree ERA5 baselines (`configs/baselines/era5`) and the
`aimip-base-troy`/`aimip-base-brian` configs. Model hyperparameters are copied
through unchanged from v4 and have not been re-tuned for the larger grid or
memory footprint. The submission overrides for the 1-degree footprint
(`N_GPUS=8`, `--shared-memory 400GiB`) live in `submit_seed_jobs.py`, not the
config.

