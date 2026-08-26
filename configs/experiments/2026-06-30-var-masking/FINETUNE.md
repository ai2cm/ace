# Multi-step fine-tuning of the paper-final var-masking runs

Continues the four paper-final v5 pre-trained checkpoints (one per
global-mean-removal x masking cell). Each fine-tune config **is that run's exact
1-step pre-training config** with only one change: `stepper_training.n_forward_steps`
is swapped from `1` to a multi-step probability schedule (plus loading the
pre-trained weights). The schedule is the only thing borrowed from the ERA5
baseline multi-step fine-tune
(`configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml`); inference
suite, training/validation windows, optimizer, loss, EMA, `max_epochs`, masking,
global-mean-removal, and architecture are all identical to pre-training.

Two configs are written per cell — **eight in total** — differing only in which
pre-training checkpoint they start from. See [Variants](#variants).

## Source checkpoints (one per cell)

| Cell | Paper label | Source pre-training run | Beaker dataset |
| --- | --- | --- | --- |
| gmroff-mask0 | No mask | `...-gmroff-mask0-seed1-v5` | `01KZEZNEAASGS4JAJKSQB192GF` |
| gmroff-mask20 | Mask 20 | `...-gmroff-mask20-seed1-v5` | `01KZEFBKGFJ2V38N8V9HNAVZ27` |
| gmron-mask0 | No mask, GMR | `...-gmron-mask0-seed1-v5` | `01KZSAQJD15697SFFCTJ1SRSA0` |
| gmron-mask20 | Mask 20, GMR | `...-gmron-mask20-seed0-v5` | `01KYT8YZZZGKGJFFK6TNJ64SFN` |

Each run's exact 1-step pre-training config (the `config.yaml` the checkpoint was
trained with) is cached under `pretrain_source_configs/`; the generator reads it,
applies the two changes below, and writes the fine-tune config. So masking,
global-mean-removal, SFNO settings, and all training/eval details are inherited
verbatim from pre-training.

## The only changes from pre-training

- `stepper_training.n_forward_steps`: `1` -> probability schedule over
  {1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05} (the ERA5-baseline multi-step
  schedule — the only borrowed piece)
- `stepper_training.parameter_init.weights_path`: added, loads the source
  checkpoint from the mounted dataset — which one depends on the variant
  ([below](#variants))
- `max_epochs`: `150` -> `20` (fine-tuning is short; set by `FT_MAX_EPOCHS`)
- **inline inference pruned**: the weight-0.0 multi-year diagnostics
  (`10year`, `10year_insample`, `long_46year`) are dropped from inline inference
  (`INLINE_INFERENCE_DROP`) — they cost hundreds-to-thousands of windows per
  inference-epoch and dominated FT wall-clock. Only `aimip_checkpoint`
  (weight 1.0, drives checkpoint selection) + the cheap `weather` entries stay.
  The dropped diagnostics run post-FT instead: the eval suite is built from the
  **unpruned** pre-training config, so they are not lost — see
  [Evaluating the fine-tunes](#evaluating-the-fine-tunes).
- `stepper.step.config.input_dropout_optimized_steps_only`: `true` — input
  masking applies only on the optimized (last) rollout step, not the
  intermediate `no_grad` steps (no-op for the mask0 cells; matters for mask20).
  Without it, masking perturbs the trajectory feeding the optimized step while
  inference runs unmasked.

Everything else is pre-training verbatim, including FusedAdam lr 1e-4,
EnsembleLoss (crps 0.9 / energy 0.1, no extra weights),
`optimize_last_step_only: true`, `n_ensemble: 2`, the full v5 inference suite,
the 1979–2008 training windows, and `logging.project: VarMasking8`.

## Variants

Pre-training writes two candidate checkpoints, selected by different criteria
(`fme/core/generics/trainer.py`), and it is not obvious which is the better
starting point for a multi-step fine-tune. Both are generated, one config each,
listed in `FT_VARIANTS` in `generate_finetune_configs.py`:

| Config/run suffix | Starts from | Selected by |
| --- | --- | --- |
| `-mstepft` | `training_checkpoints/best_ckpt.tar` | lowest validation loss; the ERA5 baseline recipe |
| `-mstepftaimip` | `training_checkpoints/best_inference_ckpt.tar` | lowest inference error, i.e. the weight-1.0 `aimip_checkpoint` entry |

The two configs are byte-identical apart from `parameter_init.weights_path` and
a header comment naming the checkpoint. In every source dataset
`best_inference_ckpt.tar` is written at an *earlier* epoch than `best_ckpt.tar`,
so `-mstepftaimip` also starts from a less-trained model.

`-mstepftaimip` matches the criterion the `-bestinf` evaluations report on,
which is why it is worth running alongside. The suffix deliberately avoids
ending in a checkpoint suffix: `update_beaker_map.py` filters names for which
`eval_checkpoints.is_derived_run_name` is true out of the run -> dataset map,
since those denote evaluation runs.

## Run

GPU count is per-cluster to avoid wasting the more powerful accelerators:
**titan (B200) uses 4 GPUs, jupiter (H100) uses 8**. `batch_size: 8` is the
global batch (local = batch_size // world_size), so 4 vs 8 GPUs trains
identically and 8 stays divisible by both. Because one beaker job requests a
fixed GPU count and could land on any allowed cluster, `submit_finetune_jobs.py`
rejects mixing clusters with different counts -- **submit one cluster at a time**.

`submit_finetune_jobs.py --variant` picks which set to submit and **defaults to
`aimip`**, not `all`. A submit writes a new `/results` and restarts fine-tuning
at epoch 0, so re-submitting a variant that is already running discards its
progress; launching the `-mstepft` set has to be an explicit `--variant best`
(or `all`).

To submit one cell rather than a whole set, name its config file(s)
positionally — that bypasses `--variant`, so it works regardless of which
variant the config belongs to.

```bash
# regenerate all eight run_configs/*-mstepft*.yaml (needs current dataset IDs)
python generate_finetune_configs.py

# dry run first (one cluster at a time)
python submit_finetune_jobs.py --dry-run \
  --beaker-cluster ai2/titan --beaker-priority high

# submit the -mstepftaimip set to titan (4 B200 GPUs / 400GiB each)
python submit_finetune_jobs.py --variant aimip \
  --beaker-cluster ai2/titan --beaker-priority high --beaker-workspace ai2/ace

# ...or just one cell, whatever its variant
python submit_finetune_jobs.py \
  ace-train-config-4deg-nc-sfno-era5-gmron-mask0-seed1-v5-mstepft.yaml \
  --beaker-cluster ai2/titan --beaker-priority high --beaker-workspace ai2/ace

# ...or submit to jupiter (8 H100 GPUs)
python submit_finetune_jobs.py --variant aimip \
  --beaker-cluster ai2/jupiter --beaker-priority high --beaker-workspace ai2/ace
```

Fine-tune run names are the source run name + the variant suffix, wandb group
`ace2-var-masking-mstepft-2026-06-30`.

## Evaluating the fine-tunes

`generate_eval_configs.py` covers the fine-tune family automatically (v5 only,
via `generate_finetune_configs.iter_train_configs`), writing one suite config
per fine-tune that has finished in wandb and has a result dataset in
`wandb_to_beaker_map.json`:

```bash
python update_beaker_map.py -v v5      # pick up finished FT result datasets
python generate_eval_configs.py -v v5  # writes ace-eval-suite-config-...-mstepft.yaml
# submit only the FT suites — naming configs positionally bypasses --version,
# so a v5 eval sweep of the pre-training runs is not dragged along
python submit_eval_jobs.py --dry-run --checkpoint lastepochema \
  ace-eval-suite-config-4deg-nc-sfno-era5-gmron-mask20-seed0-v5-mstepft.yaml
```

**Evaluate the fine-tunes at `--checkpoint lastepochema` only.** The default
runs each suite against every checkpoint of the result dataset, which is right
for the pre-training runs (reported on `-bestinf`), but a fine-tune is a fixed
`max_epochs: 20` continuation of an already-converged checkpoint: the
fine-tuned model *is* the final epoch, and `best_ckpt.tar` /
`best_inference_ckpt.tar` can only select a partially fine-tuned one. Passing
one checkpoint also cuts the pass from 4 jobs per cell to 1, each carrying the
67176-step `long_46year`.

**`lastepochema`, not `lastepoch`: the final epoch has two checkpoints and only
one of them is the model you have been looking at.** These runs set
`validate_using_ema: true`, so validation and inline inference run inside
`EMATracker.applied_params` and every number on the training run's own charts is
the EMA-averaged model. `ckpt.tar` is the restart checkpoint, saved *outside*
that context, and its EMA state lives in a separate `"ema"` key that
`load_stepper` never reads — so `-lastepoch` evaluates the raw weights.
`ema_ckpt_0020.tar` is written inside the context by `save_all_checkpoints` and
is what `lastepochema` resolves to. Measured on
`ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5-mstepft`, at the same epoch 20
and on the same data, `aimip_checkpoint/annual/rmse/air_temperature_7` is
**0.0428** inline (EMA) against **0.1818** from `-lastepoch` (raw). Note
`best_ckpt.tar` and `best_inference_ckpt.tar` are *also* saved inside the EMA
context, so `-besttrain`/`-bestinf` were EMA evaluations all along: `-lastepoch`
was the only odd one out, and a pre-training-vs-fine-tune comparison across
those two was comparing weight flavors, not training recipes.

`lastepochema` resolves its path by listing the result dataset and taking the
highest-epoch `ema_ckpt_XXXX.tar` (0020 for the fine-tunes, 0150 for
pre-training), so it needs no per-family configuration. A run with no EMA
checkpoint at all — killed before the first epoch `ema_checkpoint_save_epochs`
selects — has that one job skipped with a message, and the rest of the sweep
proceeds.

Before submitting, confirm the run actually reached epoch 20 — a wandb state of
`finished` is not proof, since `ckpt.tar` is whatever the last epoch written
was:

```python
import wandb
runs = {r.name: r for r in wandb.Api().runs("ai2cm/VarMasking8")}
r = runs["ace2-var-mask-nc-sfno-era5-gmron-mask0-seed1-v5-mstepft"]
print(r.state, r.summary["epoch"])  # want: finished 20
```

`summary["epoch"]` counts from 1, so a completed run reports exactly its
`max_epochs` (calibrated against the finished `max_epochs: 150` pre-training
runs). Skip any cell that comes up short rather than filing a partial
fine-tune alongside the others. Note `--checkpoint` takes one value per flag
and is repeated for several; it deliberately does not take `nargs="+"`, which
would swallow the positional config names.

Crucially the suite is built from the **pre-training** config, not the
fine-tune config: the fine-tune config has `INLINE_INFERENCE_DROP` applied, so
inheriting it would mean the multi-year diagnostics never run anywhere. Each
suite therefore holds all six inference entries — `aimip_checkpoint` and the
two `weather` entries kept inline, plus `10year`, `10year_insample` and
`long_46year`. Plot `-lastepochema` for the fine-tunes; the pre-training runs
stay on `-bestinf`.

The four `-mstepft` runs launched before `001a58702` applied
`INLINE_INFERENCE_DROP`, so their wandb configs still carry all six inline
entries and `long_46year/annual/*` **is** on their own charts — but only from
epochs 1 and 11, since those entries run on an `epochs: {start: 0, step: 10}`
schedule that never lands on epoch 20. That is the second reason an inline
number and its `-lastepoch` counterpart disagreed: different epoch on top of
different weight flavor. Fine-tunes generated after that commit have the
multi-year entries pruned and get them only from the eval suite.

Regenerating an eval pass for a pre-v5 family means passing `--checkpoint
bestinf` explicitly. Those families were evaluated when the default was three
checkpoints wide; taking today's default would submit a `-lastepochema` job for
every cell.

## Memory

The multi-step rollout of the embed_dim-512 var-masking model (with channel-mask
inputs + GMR) is heavier than the ERA5 baseline this schedule came from and can
fragment GPU memory near the 80 GiB H100 limit. `run-ace-train.sh` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (overridable) to reclaim
reserved-but-unallocated memory. If a run still OOMs, escalate with gradient
checkpointing in the builder (`stepper.step.config.builder.config.checkpointing`,
1 = encoder/decoder … 3 = per-block; reproducibility-safe, ~20-30% slower).
