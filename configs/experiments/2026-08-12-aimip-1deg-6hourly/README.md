# AIMIP-like 1°, 6-hourly

6-hourly counterpart of the 1°/daily v2 ERA5-only no-residual no-CO2 baseline submitted to
AIMIP (`train-1deg-daily-v2-era5-only-no-residual-no-co2.yaml`, branch
`experiment/2026-07-15-1deg-daily-v2-era5-only-no-residual-no-co2`).

A daily-step model emits daily snapshots, which cannot be evaluated against the monthly- and
daily-AVERAGED AIMIP ERA5 target. A 6-hourly step gives four snapshots per day, which average
into comparable means.

## Stages

| stage | config | wandb | result dataset |
|---|---|---|---|
| 1. pretrain (1-step) | `…-no-co2.yaml` | `g94277n6` | `01KZYJ4HTBWED5VG3VFTRYKDRC` |
| 2. multi-step FT | `…-multi-step-ft.yaml` | `78crdqjr` | `01M0RFP2DKAGABV89KRPMXX5C3` |
| 3. pressure-level FT | `…-plev-ft.yaml` | `lmvpfmrp` | — |

Each stage mounts the previous stage's result dataset at `/weights` via its `# arg:` header.
Stage 2 initializes from `best_ckpt.tar`, stage 3 from `best_inference_ckpt.tar`.

## Changes from the daily config (stage 1)

- **Data**: `data_path: /climate-default/` with the unchanged `file_pattern` — the 6-hourly
  store has no wrapper directory. Train/val windows are calendar ranges, unchanged, split on
  the ERA5 stream-stitch boundaries (1986-04-01, 1993-08-01).
- **Stats**: `-daily-stats-` → `-stats-`. Residual scaling is timestep-dependent, so the daily
  stats cannot be reused.
- **Horizons ×4** at fixed lead time. `forward_steps_in_memory` is an IO chunk size, not a
  horizon, and is unchanged.
- `max_epochs` 60 → 40; loader `num_data_workers` 8 → 4; train `prefetch_factor` 4 → 2.

Initial conditions stay at 06Z, which exists in the 6-hourly store (00/06/12/18Z).

## Stage 3: pressure-level fine-tune

Adds `ta`/`hus`/`ua`/`va` at the 7 AIMIP levels (1000/850/700/500/250/100/50 hPa) as a
`secondary_decoder` MLP head, trunk frozen via
`parameter_init.parameters: [{frozen: {include: ["*"]}}]` (~20k trainable parameters).

- **No plev dataset is needed.** The 6-hourly store carries the pressure-level fields and the
  existing stats cover them. Only the *daily* store drops levels, which is why the daily
  workflow had to build one.
- **27 names, not 28.** `secondary_diagnostic_names` may not overlap `in_names`/`out_names`
  (raises at stepper build), and `TMP850` is already an `out_name`.
- Built from the **stage 1** config, because `stepper:` must be restated in full:
  `CheckpointStepperConfig` accepts only `checkpoint_path` and cannot carry
  `secondary_decoder`.
- `max_epochs: 50` is a ceiling. Stop when `best_inference_error` stops improving.

## Do not select on 2015-2024

That is the AIMIP holdout period: no training, validation, or checkpoint selection. The
`10year` inference on 2015 ICs carries `weight: 0.0` for this reason — it is look-only, and
giving it weight would violate the protocol. The weighted rollout is `10year_insample`
(1995-2004), which lies inside the training window.

## Launching

```bash
cd configs/experiments/2026-08-12-aimip-1deg-6hourly
./run-train.sh
```

Stages are `run_training` calls; uncomment the one to run and comment out the others. Put the
preceding stage's result dataset id in the config's `# arg:` header — it must be committed
(job finalized) before beaker will mount it. `N_GPUS` is only correct for the cluster the
launcher pins.
