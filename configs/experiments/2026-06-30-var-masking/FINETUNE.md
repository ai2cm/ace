# Multi-step fine-tuning of the paper-final var-masking runs

Continues the four paper-final v5 pre-trained checkpoints (one per
global-mean-removal x masking cell) with the **exact multi-step fine-tuning
recipe of the ERA5 baseline**
(`configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml`).

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
`generate_finetune_configs.py`, `update_beaker_map.py`, re-generate, re-submit.

The model architecture is reconstructed per cell from each checkpoint
(`stepper.checkpoint_path`), so masking / global-mean-removal / SFNO settings are
inherited exactly from pre-training; `stepper_training.parameter_init` loads the
same weights as the fine-tuning start.

## Fine-tuning recipe (copied from the ERA5 baseline)

- `stepper_training.n_forward_steps`: probability schedule over
  {1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}
- `optimize_last_step_only: true`, `n_ensemble: 2`
- `loss`: EnsembleLoss (crps 0.9 / energy 0.1), `h500` weight 5.0
- `optimization`: AdamW, lr 1e-4, weight_decay 0.01, fused, grad accumulation, amp off
- `max_epochs: 40`, EMA decay 0.999, validate using EMA
- inference / train / val windows: verbatim from the baseline FT config
- checkpoint file loaded: `training_checkpoints/best_ckpt.tar`

**Only deviations from the baseline FT config:** `logging.project: VarMasking8`
(so fine-tunes group with the experiment and the eval tooling finds them), and
the per-cell `/weights` checkpoint mount.

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
