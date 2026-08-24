# AIMIP-like 1°, 6-hourly step

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

## Three-stage training

Stages 1-2 mirror the ACE2S pair in `configs/baselines/era5/`; stage 3 follows the
pressure-level fine-tune in `configs/experiments/2026-05-19-era5-aimip-ace2s/`.

| stage | config | `n_forward_steps` | status |
|---|---|---|---|
| 1. pretraining | `…-no-co2.yaml` | `1` | done — wandb `g94277n6`, beaker `01KZYJ4HT4ZMZH296KBNWMPCQF`, result dataset `01KZYJ4HTBWED5VG3VFTRYKDRC` |
| 2. multi-step FT | `…-no-co2-multi-step-ft.yaml` | stochastic 1/2/4/12/20 @ 0.6/0.2/0.1/0.05/0.05 | done — wandb `78crdqjr`, beaker `01M06W0WN4WY1HJBWFXJNJEXC7`, result dataset `01M0RFP2DKAGABV89KRPMXX5C3` |
| 3. pressure-level FT | `…-no-co2-plev-ft.yaml` | `1` (frozen trunk, plev head only) | not launched |

Stage 1 ran at commit `b4a688d85` and finished 40 epochs in 44.5 h on 4 GPUs, with
`best_val_loss` 0.1271 still falling at the final epoch — so the fine-tune's donor is a
still-improving checkpoint rather than a converged one.

## Stage 2: multi-step fine-tune

Completed 2026-08-24: 40 epochs, 248,400 batches, 81 h wall across 35 preempted attempts
(Ai2's scheduler toggles high/urgent, so preemption is routine and training resumed cleanly
each time — beaker repopulates the results dir on retry). Final `best_val_loss` 0.16528;
`best_inference_error` 0.026124, **unchanged since epoch 8**.

Against stage 1 the gains were large and mostly immediate: rollout error 0.04533 -> 0.02706
by the fine-tune's 4th epoch (−40%), and the precipitation small-scale power deficit
0.44 -> 0.33 over the same span. Roughly 78-80% of the total spectral gain is in place by
epoch 8, with ~20% accruing over epochs 8-36; the rollout-error gain is fully in place by
epoch 8.


The fine-tune config is the pretrain config with six changes and nothing else:

- **Donor checkpoint.** A `# arg: --dataset <id>:/weights` header mounts the pretrain job's
  result dataset; `stepper_training.parameter_init.weights_path` and
  `stepper.checkpoint_path` both point at `/weights/training_checkpoints/best_ckpt.tar`.
  The two are complementary — `checkpoint_path` supplies the stepper *config*
  (architecture, normalization, corrector, `global_mean_removal`, in/out names),
  `parameter_init` supplies the *weights*. `CheckpointStepperConfig` has exactly one field
  and parsing is strict, so nothing under `stepper:` can be overridden; the recipe is
  inherited wholesale, which is the intent. Note `parameter_init` does not restore EMA
  state, so the fine-tune builds a fresh EMA tracker from the loaded weights.
- **`best_ckpt.tar`, not `ckpt.tar`.** The pretrain ran `validate_using_ema: true`, so the
  best-validation checkpoint is saved inside the EMA context and holds EMA-averaged
  weights.
- **Rollout distribution** as above, with `optimize_last_step_only: true` (already set in
  the pretrain config): only the sampled last step carries gradients, earlier rollout steps
  run under `torch.no_grad()`.
- **`validation.evaluate_all_steps: false`** and validation `batch_size` 32 → 16. The
  20-step outcome sizes *every* data window at 21 timesteps regardless of the sampled
  length, so the default (`true`) would roll out 20 steps on every validation batch —
  roughly 20× the pretrain's 127 s/epoch, about 28 h over 40 epochs. The trade-off is that
  each `loss_step_N` is averaged over only the batches that sampled more than N steps.
- **10-year inference cadence** halved, `{start: 1, step: 2}` → `{start: 3, step: 4}`, i.e.
  epochs 4, 8, …, 40 instead of every second epoch. In the pretrain run the `10year` pair
  was 73.5% of all inference cost (~53,400 s of 72,628 s) against 25.8% for `long_46year`,
  so this is the largest single lever. `long_46year` keeps epochs 30/35/40.
- Everything else — `max_epochs: 40`, `seed`, `lr`, `ema.decay`, the training loader, and
  all inference horizons and initial conditions — is unchanged, as in every other
  fine-tune/pretrain pair in the repo.

## Stage 3: pressure-level fine-tune

Adds the AIMIP evaluation pressure-level diagnostics — `ta`/`hus`/`ua`/`va` at
1000/850/700/500/250/100/50 hPa — as a `secondary_decoder` MLP head, with the donor trunk
frozen. Derived from the **stage 1** config (not stage 2), because `stepper:` must be
restated in full: `CheckpointStepperConfig` takes only `checkpoint_path` and strict parsing
forbids adding `secondary_decoder` beside it. Stage 2 inherited stage 1's stepper config
verbatim, so restating it reproduces the stage-2 architecture.

**No plev dataset is needed.** The daily workflow had to build one
(`origin/plev-daily-coarsen-and-window-fix`) because the *daily* store keeps only a few
pressure levels. The 6-hourly store carries all of them — `TMP`/`Q`/`UGRD`/`VGRD` at
10/50/100/200/250/500/700/850/1000 hPa and `h` additionally at 300 — and
`2026-03-19-era5-1deg-8layer-stats-1990-2019` already covers every plev channel (134
variables, verified). The stale `2025-11-10-era5-1deg-pressure-level-1940-2022.zarr` from the
ACE2.1 workflow is not used, and the ACE2.1 13-level set (150/300/400/600/925) is not
reproduced.

**27 names.** Four variables (`ta`/`hus`/`ua`/`va` -> `TMP`/`Q`/`UGRD`/`VGRD`) at the 7 AIMIP
levels is 28, less `TMP850` which is already in `out_names` —
`secondary_diagnostic_names` may not overlap `in_names`/`out_names`
(`fme/core/step/single_module.py`). This matches the ACE2.2 daily plev fine-tune's set
exactly. Geopotential height (`zg`/`h`) is deliberately excluded: the AIMIP levels are
specified for ta/hus/ua/va, and the trunk already emits `h500` natively.

**Donor is stage 2's `best_inference_ckpt.tar`, not `best_ckpt.tar`.** The inference-error
criterion is the one prior checkpoint selection used, and on stage 2 it peaked at **epoch 8**
(0.026124) and never improved through epoch 40 — the composite is dominated by time-mean
terms, which drifted while short-lead skill and precipitation spectra kept improving. Taking
epoch 8 keeps selection consistent with the earlier AIMIP checkpoints rather than optimising
a different metric mid-experiment. It is EMA-wrapped, since stage 2 ran
`validate_using_ema: true`.

Note the donor dataset is the result dataset of stage 2's **final** job. Beaker repopulates
the results directory on every preemption retry, so each job's dataset carries the full
checkpoint set — the epoch-8 `best_inference_ckpt.tar` is present in the last one, with a
timestamp reflecting the copy-forward rather than when it was computed.

**Train/validation split: unchanged across all three stages.** Train 1979-1993 + 1995-2013
(stitch-aligned blocks), validate 1994 + 2014 — inherited verbatim from the ACE2.2 daily
training config that stage 1 ports.

This differs from the ACE2.2 *pressure-level* fine-tune (wandb `gvoj4hf7`), which used the
plev-FT template's split: train 1979-2008, validate 2009-2014 with `step: 3`. That template
traces back to the ACE2.1-era aimip configs and carries its own split regardless of donor, so
the ACE2.2 lineage changed protocol between its training run and its plev fine-tune. Ours does
not. Both splits are AIMIP-compliant.

Consequence: this head's loss is **not** directly comparable to `gvoj4hf7`'s, because the
validation years differ. Retraining stages 1-2 on the plev-template split was considered and
deferred — it would buy cross-run comparability and remove a protocol change that is awkward
to explain in an AIMIP publication, but costs a full re-run of both stages.

**The inference cadence is deliberately dense** (`{start: 1, step: 2}`, every 2nd epoch). The
first attempt used every 10th epoch and its first evaluation landed at epoch 10, by which
point the head had already converged — at 6210 batches/epoch, one epoch here is ~9x the
optimizer steps of the daily ACE2.2 plev FT's, so the whole interesting phase happens in the
first few epochs.

**`max_epochs: 50` is an upper bound, not a target.** The prior plev fine-tunes were all
early-stopped on convergence rather than run to their epoch limit, and how long that takes
varies a lot — one converged by epoch 41, another was still worth running at 73. Watch
`best_val_loss` and stop when it goes flat; the prior runs all fell below ~0.5% improvement
over their final quarter. Stopping early costs nothing, since `best_ckpt.tar` plus
`ema_ckpt_XXXX` every 5 epochs are already saved.

Other changes from stage 1: `max_epochs` 40 -> 50; `stepper_training` becomes single-step
`MSE` with `parameter_init.parameters: [{frozen: {include: ["*"]}}]`; and the inference set
collapses to `10year_insample` alone, at every 2nd epoch — the same weighted rollout stages
1-2 select on, so `best_inference_ckpt` means the same thing at every stage.

### Do not select on 2015-2024

**2015-2024 is the AIMIP holdout period.** It must not be used for training, validation, or
checkpoint selection. The `10year` inference in stages 1-2 runs on 2015 ICs and carries
`weight: 0.0` *for this reason* — it is a look-only diagnostic, not a selection signal. Giving
it a non-zero weight would violate the protocol. Stage 3 omits it entirely.

Consequence: the weighted rollout (`10year_insample`, 1995-2004) lies **inside** the training
window, so the inference-based criterion is not held-out. That is forced by the split — with
training through 2013, validation on 1994/2014, and 2015+ embargoed, no non-holdout
out-of-sample decade remains. ACE2.1 could select out-of-sample only because its split
reserved 2009-2014 for validation, which our stages 1-2 train on.

Mitigating factor: a decade-long free rollout decorrelates from its initial condition within
weeks, so what is scored is the model climate rather than a fit to specific years. Note also
that `best_ckpt.tar` selects on held-out validation years (1994/2014) regardless. A frozen trunk means the
prognostic rollout is unchanged from stage 2, so the `weather_*` and `long_46year`
diagnostics would re-measure a model that cannot have moved.

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
cd configs/experiments/2026-08-12-aimip-1deg-6hourly
./run-train.sh
```

Both stages are `run_training` calls in `run-train.sh`, in reproduction order: stage 1 is
live, stage 2 is commented out. To reproduce from scratch, run stage 1, take its beaker
result dataset id, put it in the fine-tune config's `# arg:` header, then uncomment stage
2. There is no automation for that lookup anywhere in the repo, so the id is pasted in by
hand; the ids from our own stage 1 run are recorded next to the commented-out stage 2 call.

Note that stage 1 has already run, so launching the script as-is relaunches a 44-hour job.

Targets and GPU count live in `run-train.sh`. `N_GPUS` is only correct for the cluster the
launcher pins, since per-GPU memory is cluster-specific — retarget by changing both
together.

`batch_size: 8` is the *global* batch; `dist.local_batch_size` divides it by the rank
count, so the effective batch and the gradient are independent of `N_GPUS`.
