# Baseline config versions

- `ace2-var-mask-nc-sfno-era5-v1.yaml`: original var-masking baseline.
  `global_mean_co2` is an input channel, listed in both
  `stepper.step.config.next_step_forcing_names` and
  `stepper.step.config.in_names`.
- `ace2-var-mask-nc-sfno-era5-v2.yaml`: same as v1 but with `global_mean_co2`
  removed from `next_step_forcing_names` and `in_names` (no longer a network
  input). The `long_46year_constant_co2` inference entry's
  `persistence_names: [global_mean_co2]` is unchanged in both versions.
