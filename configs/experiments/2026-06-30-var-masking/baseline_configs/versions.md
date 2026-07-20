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
