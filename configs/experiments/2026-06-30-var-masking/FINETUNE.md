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
  Run the dropped climate diagnostics post-FT via the eval tooling.
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
ending in `-bestinf`: `update_beaker_map.py` filters names ending in
`-bestinf`/`-besttrain`/`-lastepoch` (`SKIP_SUFFIXES`) out of the run -> dataset
map, since those denote evaluation runs.

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
`ace2-var-masking-mstepft-2026-06-30`. Evaluate them with the existing eval
tooling (`generate_eval_configs.py` / `submit_eval_jobs.py`) once trained.

## Memory

The multi-step rollout of the embed_dim-512 var-masking model (with channel-mask
inputs + GMR) is heavier than the ERA5 baseline this schedule came from and can
fragment GPU memory near the 80 GiB H100 limit. `run-ace-train.sh` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (overridable) to reclaim
reserved-but-unallocated memory. If a run still OOMs, escalate with gradient
checkpointing in the builder (`stepper.step.config.builder.config.checkpointing`,
1 = encoder/decoder … 3 = per-block; reproducibility-safe, ~20-30% slower).
