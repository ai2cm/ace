# AIMIP-like 1° baseline, 6-hourly step

`train-1deg-6hourly-v2-era5-only-no-residual-no-co2.yaml` is a 6-hourly-timestep
counterpart of the 1°/daily v2 ERA5-only no-residual no-CO2 baseline submitted to AIMIP,
which trained on the daily-mean-timestep ERA5 store
(`train-1deg-daily-v2-era5-only-no-residual-no-co2.yaml`, branch
`experiment/2026-07-15-1deg-daily-v2-era5-only-no-residual-no-co2`).

## Why

A daily-step model emits a sequence of daily *snapshots*. The AIMIP ERA5 evaluation
dataset is monthly- and daily-**averaged**, so it samples the full diurnal cycle — a
snapshot series cannot be compared against it, which is what made the daily model
incompatible with the AIMIP evaluation. A 6-hourly step gives four snapshots per day, which average into
genuine daily/monthly means directly comparable to the evaluation target.

## What changes from the daily config, and nothing else

- **Data store.** Only `data_path` changes:
  `/climate-default/2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr` →
  `/climate-default/`. The daily store is a wrapper directory containing a zarr named
  `2026-03-19-era5-1deg-8layer-1940-2025.zarr`; the 6-hourly store *is* that zarr, sitting
  at the mount root with no wrapper, so the unchanged `file_pattern` selects it. Applies
  to all 13 dataset blocks (5 inference loaders, 6 train concat entries, 2 validation
  concat entries). The `subset:` windows are calendar ranges and are unchanged, including
  the ERA5 production-stream stitch boundaries (1986-04-01, 1993-08-01) the training
  concat is split on.

- **Normalization.** `2026-03-19-era5-1deg-8layer-daily-stats-1990-2019` →
  `2026-03-19-era5-1deg-8layer-stats-1990-2019` (all four paths). Residual scaling is
  timestep-dependent, so the daily stats cannot be reused. Equivalent beaker dataset:
  `andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019`.

- **`max_epochs`: 60 → 40**, matching the 6-hourly ACE2S-ERA5 baseline
  (`configs/baselines/era5/ace-train-config-1-step-pretrain.yaml`). No LR scheduler is
  configured, so the constant-LR schedule is unaffected by the epoch count. Both epoch
  cadences are index slices into the epoch list and still land on the final epoch at 40,
  so neither needed editing: `ema_checkpoint_save_epochs` (`start: 5, step: 5`) →
  `range(41)[5::5]` = 5, 10, …, 40; the 10-year inferences (`start: 1, step: 2`) →
  `list(range(1, 41))[1::2]` = 2, 4, …, 40.

- **Horizons ×4 at fixed lead time.** `10year` and `10year_insample` 3652 → 14608; the
  5-day weather evals 5 → 20 (`n_forward_steps` and `forward_steps_in_memory`, which
  equals the horizon there); `long_46year` 16794 → 67176; every aggregator
  `step_means`/`ensembles` `step:` index 5 → 20. `forward_steps_in_memory: 40` on the
  10-year and 46-year runs is a memory/IO chunk size, not a horizon, and is unchanged.
  Aggregator names stay `day_5*` — they are still day 5.

- **`long_46year` cadence.** At 67176 steps this rollout is 4× its daily cost, so it runs
  at the end rather than throughout: `epochs: {start: 4, step: 5}` → `{start: 29, step: 5}`
  = epochs 30, 35, 40. It keeps `weight: 0.0` (diagnostic only).

- **Loader concurrency.** `num_data_workers: 8 → 4` on all seven loaders and
  `train_loader.prefetch_factor: 4 → 2` — the settings measured on the 6-hourly
  `stochastic-ace-bakeoff-6h` wave, which cut rank-0 RSS from ~113 GiB to ~57 GiB and
  ended the host-RAM OOM/wedge failures that killed several 6-hourly runs. `time_buffer`
  is unchanged (train 1, validation 3).

Initial-condition timestamps stay at `T06:00:00`: 06Z exists in the 6-hourly store
(00/06/12/18Z), and keeping it makes lead times directly comparable with the daily runs.

## Held fixed deliberately

`seed: 0`, batch sizes (8 train / 32 validation), `optimization` (FusedAdam, lr 1e-4,
weight decay 0.01, gradient accumulation), `ema.decay: 0.999`, `stepper_training`
(`n_ensemble: 2`, 1 forward step, EnsembleLoss crps 0.9 / energy score 0.1), and the whole
`stepper` block — `residual_prediction: false`, NoiseConditionedSFNO with
`filter_num_groups: 16` and `spectral_ratio: 0.125`, `ocean`, `corrector`,
`global_mean_removal` (`kind: shared`, `append_as_input: true`), and the
`in_names`/`out_names`/`next_step_forcing_names` lists (no `global_mean_co2`, per the
AIMIP protocol).

Consequences to be aware of: 40 epochs at a 6h step is ~2.7× the optimizer steps and wall
time of the 60-epoch daily run, per-epoch EMA shapes differ in step terms, and each
inference costs ~4× its daily counterpart.

## Launching

```bash
cd configs/baselines/aimip-like-6hourly
./run-train.sh
```

Targets and GPU count live in `run-train.sh`. `N_GPUS` is only correct for the cluster the
launcher pins, since per-GPU memory is cluster-specific — retarget by changing both
together.

`batch_size: 8` is the *global* batch; `dist.local_batch_size` divides it by the rank
count, so the effective batch and the gradient are independent of `N_GPUS`.
