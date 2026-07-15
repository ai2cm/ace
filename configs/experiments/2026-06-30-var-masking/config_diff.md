# Config Diff: Old vs New

- Old: `VarMasking4/ace2-var-mask-nc-sfno-mask20-uniform-co2-default-v4.yaml`
- New: `VarMaskingERA5/ace2-var-mask-nc-sfno-era5-mask30-co2default-v2.yaml`

## Masking scheme (main point of comparison)

- Old: `input_dropout: {kind: uniform, min_vars: 0, max_vars: 20}` — uniform random 0-20 vars masked.
- New: `input_dropout: {default: {max_masked_vars: 30}}` — new masking config shape, fixed 30 max masked vars, no `kind`/`min_vars`.

## EMA / checkpointing

- Old: no `ema_checkpoint_save_epochs`.
- New: adds `ema_checkpoint_save_epochs: {start: 10, step: 10}`.

## Train aggregator

- New adds top-level `train_aggregator: {ensemble_metrics: true}`. Old has none.

## Inference suite

- Old inference epochs `{start: 0, step: 5}` (or `{start:0, step:2}` for 10year); New uses `{start: 10, step: 10}` uniformly, and adds it to `weather_2024`/`weather_1994`/`long_46year` blocks (Old lacked epochs there).
- Old `10year` run: weight 1.0, dates from 2015, name `10year`.
- New splits this into two runs: `10year` (weight 0.0, from 2015) and `10year_insample` (weight 1.0, from 1995).
- Old has separate `long_46year_constant_co2` (weight 0.0, first block) and `long_46year` (weight 0.0, last block).
- New has both too, but order flipped (`long_46year` before `long_46year_constant_co2`), same weights/content, epochs added.
- `num_data_workers` for inference loaders: Old 8, New 4.
- `day_5` aggregator variable list (`id001`): New adds `air_temperature_2/4/5/6`, `specific_total_water_2/4/5/6`, `eastward_wind_2/4/5/6`, `northward_wind_2/4/5/6`, and `total_water_path`. Old only has levels 0/1/3/7 for those fields (no `total_water_path`).

## Logging

- `project`: Old `VarMasking4` → New `VarMaskingERA5`.

## train_loader / dataset concat ranges

- Old: 2 concat blocks, `1979-01-01..1993-12-31` and `1995-01-01..2013-12-31`.
- New: 6 concat blocks, splitting into finer subsets: `1979-01-01..1986-03-31`, `1986-04-01..1993-07-31`, `1993-08-01..1993-12-31`, `1995-01-01..1999-12-31`, `2000-01-01..2009-12-31`, `2010-01-01..2013-12-31`. Same overall span, more chunks.

## validation loader

- `num_data_workers`: Old 8 → New 4. Same date subsets otherwise.

## optimization

- Old has explicit `scheduler` block (LinearLR → ConstantLR → PolynomialLR, milestones `[8, 142]`).
- New has no `scheduler` block at all.

## stepper_training.loss

- Old: `weights` block present (per-variable loss weights: air_temperature_0, h500, TMP850, etc.).
- New: no `weights` block — just `kwargs: {crps_weight: 0.9, energy_score_weight: 0.1}`.

## stepper builder (NoiseConditionedSFNO)

- Old: `filter_num_groups: 1`, no `spectral_ratio`, no `clip_latent_global_means`.
- New: `filter_num_groups: 16`, adds `spectral_ratio: 0.125`, adds `clip_latent_global_means: true`.
- Notes: GMR off (A/B), clipping off

## stepper in_names / out_names

- New moves `global_mean_co2` from end of `in_names` to right after `HGTsfc`; also New's `in_names` no longer lists `global_mean_co2` at the end (dedup — appears once, earlier).
- Old's `in_names` has `global_mean_co2` only at the end. New has it right after `HGTsfc` and not duplicated at end.
- `out_names` unchanged in content between Old/New.

## pre_cooldown_checkpoint_epoch

- Old: `pre_cooldown_checkpoint_epoch: 142`.
- New: field absent.
