# Config version overview

Notes on what changes between `v1`/`v2`/`v3` (and other version-suffixed
variants) for each base config family in this directory. Configs with only
one version (`ace-train-config-4deg-AIMIP-sfno.yaml`,
`ace-train-config-4deg-AIMIP-nc-sfno-fm-0.1-v1.yaml`,
`ace-train-config-4deg-AIMIP-nc-sfno-fm-0.9-v1.yaml`,
`one-step-pre-train-config-full.yaml`) are not covered since there's nothing
to compare.

## `ace-train-config-4deg-AIMIP-nc-sfno-{v1,v2,v3}.yaml`

- **v1**: baseline. Single continuous ERA5 zarr dataset (no stitch splitting).
  Custom per-variable loss `weights` block. `residual_prediction: false`,
  `filter_num_groups: 1`, no `spectral_ratio`, no
  `clip_latent_global_means`. No `ema_checkpoint_save_epochs`, no
  `train_aggregator.ensemble_metrics`.
- **v2**: adds `ema_checkpoint_save_epochs` (start 5, step 5) and
  `train_aggregator: {ensemble_metrics: true}`. Adds a new
  `10year_insample` inference block and shifts several inference block
  start epochs. Splits the ERA5 training dataset into multiple `subset`
  entries at production-stream stitch boundaries (comment explains this
  avoids sampling a 1-step finite-difference target across a
  resolution-independent bias jump at a stitch). Drops the custom loss
  `weights` block (falls back to default weighting). Flips
  `residual_prediction: true`, raises `filter_num_groups` from 1 to 16,
  adds `spectral_ratio: 0.125` and `clip_latent_global_means: true`.
- **v3**: identical to v2 except `seed: 1` (vs `seed: 0`). A repeat/ensemble
  run of the same config.

## `ace-train-config-4deg-AIMIP-nc-sfno-c96-{v1,v2}.yaml`

- **v1** vs **v2**: only difference is `seed: 0` vs `seed: 1`. Pure repeat
  run, no config changes.

## `ace-train-config-4deg-AIMIP-nc-sfno-fm-random-{v1,v2,v3}.yaml`

- **v1**: baseline. Full, un-pruned training dataset — long continuous
  windows per source (e.g. AMIP ensemble spans 1939-10-02 to 2021-12-16
  across two `ic_000{1,2}` files; ramped-SST random-CO2 members include
  extra `ic_0003` duplicates; SOM members include `ic_0002`–`ic_0005` for
  each of 1x/2x/3x/4xCO2). Custom loss `weights` block.
  `residual_prediction: false`, `filter_num_groups: 1`, no
  `spectral_ratio`/`clip_latent_global_means`.
- **v2**: prunes/shortens the training dataset (e.g. AMIP window cut to
  1979-01-01–2008-12-31 and dropped to a single `ic_0001` file; fewer
  ramped-SST and SOM ensemble members/files). Drops the custom `weights`
  block. Flips `residual_prediction: true`, `filter_num_groups: 16`,
  adds `spectral_ratio: 0.125` and `clip_latent_global_means: true`.
- **v3**: reverts the dataset back to v1's full set of windows/members and
  restores the custom `weights` block, but **keeps** v2's model-side
  changes (`residual_prediction: true`, `filter_num_groups: 16`,
  `spectral_ratio: 0.125`, `clip_latent_global_means: true`). Effectively
  "v1 dataset + v2 architecture."

### `ace-train-config-4deg-AIMIP-nc-sfno-fm-random-v2-mask10-co2bern80.yaml`

- Identical to `fm-random-v2`, plus an `input_dropout` block:
  `default.max_masked_vars: 10`, and an `override_groups` entry that masks
  `global_mean_co2` at `rate: 0.8` (Bernoulli dropout). Also sets
  `include_channel_mask_inputs: true`. Tests random input-channel masking
  with CO2 dropped out at high (80%) probability.

## `ace-train-config-4deg-AIMIP-nc-sfno-fm-0.5-{v1,v2,v3}.yaml`

- **v1**: baseline. All inference block `epochs.start` values are `0`
  (evaluation starts immediately). `10year_insample_ensemble_constant_co2`
  has `weight: 0.0` (disabled). Full dataset windows/members (same style
  as fm-random v1), `group_weights.groups: [28, 2]` (28 non-ERA5 + 2 ERA5
  members), `num_data_workers: 8` for the group-weighted loader, custom
  loss `weights` block, `residual_prediction: false`,
  `filter_num_groups: 1`, no `spectral_ratio`/`clip_latent_global_means`.
- **v2**: delays inference — all `epochs.start` values shift `0 → 10`.
  Enables `10year_insample_ensemble_constant_co2` (`weight: 0.0 → 1.0`).
  Prunes the dataset the same way as fm-random v2 (shorter windows, fewer
  members), `group_weights.groups: [10, 2]`, `num_data_workers: 8 → 4`.
  Drops the custom `weights` block. Flips `residual_prediction: true`,
  `filter_num_groups: 16`, adds `spectral_ratio: 0.125` and
  `clip_latent_global_means: true`.
- **v3**: reverts the schedule/dataset/weighting changes back to v1
  (`epochs.start` back to 0, `10year_insample_ensemble_constant_co2`
  disabled again, full dataset restored, `groups: [28, 2]`,
  `num_data_workers: 8`, custom `weights` block restored) but **keeps**
  v2's model-side changes (`residual_prediction: true`,
  `filter_num_groups: 16`, `spectral_ratio: 0.125`,
  `clip_latent_global_means: true`). Same "v1 config + v2 architecture"
  pattern as fm-random v3.
