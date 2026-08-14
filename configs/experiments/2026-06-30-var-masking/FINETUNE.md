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

## Source checkpoints (one per cell)

| Cell | Paper label | Source pre-training run | Beaker dataset |
| --- | --- | --- | --- |
| gmroff-mask0 | No mask | `...-gmroff-mask0-seed1-v5` | `01KZEZNEAASGS4JAJKSQB192GF` |
| gmroff-mask20 | Mask 20 | `...-gmroff-mask20-seed1-v5` | `01KZEFBKGFJ2V38N8V9HNAVZ27` |
| gmron-mask0 | No mask, GMR | `...-gmron-mask0-seed2-v5` | `01KYWXQH0XT66EHXVRJ5N9HC4S` |
| gmron-mask20 | Mask 20, GMR | `...-gmron-mask20-seed0-v5` | `01KYT8YZZZGKGJFFK6TNJ64SFN` |

`gmron-mask0` uses **seed2** as an interim stand-in: the paper's intended
`gmron-mask0-seed1` had no succeeded checkpoint at generation time (still running
in urgent). Swap it once seed1 finishes — edit `SELECTED_SOURCES` in
`generate_finetune_configs.py`, cache its `config.yaml` into
`pretrain_source_configs/`, re-generate, re-submit.

Each run's exact 1-step pre-training config (the `config.yaml` the checkpoint was
trained with) is cached under `pretrain_source_configs/`; the generator reads it,
applies the two changes below, and writes the fine-tune config. So masking,
global-mean-removal, SFNO settings, and all training/eval details are inherited
verbatim from pre-training.

## The only changes from pre-training

- `stepper_training.n_forward_steps`: `1` -> probability schedule over
  {1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05} (the ERA5-baseline multi-step
  schedule — the only borrowed piece)
- `stepper_training.parameter_init.weights_path`: added, loads
  `/weights/training_checkpoints/best_ckpt.tar`
- `max_epochs`: `150` -> `20` (fine-tuning is short; set by `FT_MAX_EPOCHS`)

Everything else is pre-training verbatim, including FusedAdam lr 1e-4,
EnsembleLoss (crps 0.9 / energy 0.1, no extra weights),
`optimize_last_step_only: true`, `n_ensemble: 2`, the full v5 inference suite,
the 1979–2008 training windows, and `logging.project: VarMasking8`.

## Run

GPU count is per-cluster to avoid wasting the more powerful accelerators:
**titan (B200) uses 4 GPUs, jupiter (H100) uses 8**. `batch_size: 8` is the
global batch (local = batch_size // world_size), so 4 vs 8 GPUs trains
identically and 8 stays divisible by both. Because one beaker job requests a
fixed GPU count and could land on any allowed cluster, `submit_finetune_jobs.py`
rejects mixing clusters with different counts -- **submit one cluster at a time**.

```bash
# regenerate the four run_configs/*-mstepft.yaml (needs current dataset IDs)
python generate_finetune_configs.py

# dry run first (one cluster at a time)
python submit_finetune_jobs.py --dry-run \
  --beaker-cluster ai2/titan --beaker-priority high

# submit to titan (4 B200 GPUs / 400GiB each, via run-ace-train.sh)
python submit_finetune_jobs.py \
  --beaker-cluster ai2/titan --beaker-priority high --beaker-workspace ai2/ace

# ...or submit to jupiter (8 H100 GPUs)
python submit_finetune_jobs.py \
  --beaker-cluster ai2/jupiter --beaker-priority high --beaker-workspace ai2/ace
```

Fine-tune run names are the source run name + `-mstepft`, wandb group
`ace2-var-masking-mstepft-2026-06-30`. Evaluate them with the existing eval
tooling (`generate_eval_configs.py` / `submit_eval_jobs.py`) once trained.
