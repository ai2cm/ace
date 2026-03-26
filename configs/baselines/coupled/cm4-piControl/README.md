# SamudrACE training and evaluation for CM4 piControl

Self-contained baseline configs and scripts for the SamudrACE coupled
atmosphere-ocean training pipeline on the 200-year GFDL CM4 piControl data.

## Pipeline overview

The pipeline first trains uncoupled atmosphere and ocean models independently,
then couples them in two stages: ocean-only fine-tuning (`train.sh`) and then
joint ocean-and-atmosphere fine-tuning (`finetune.sh`).

```
Uncoupled training:
  uncoupled-atmos/train.sh   -> atmos checkpoint
  uncoupled-ocean/train.sh   -> ocean checkpoint

Uncoupled evaluation:
  uncoupled-atmos/evaluate.sh
  uncoupled-ocean/evaluate.sh

Coupled training stage 1:
  train.sh  -> coupled checkpoint (atmos frozen, ocean fine-tuned)

Coupled training stage 2:
  finetune.sh  -> refined coupled checkpoint (both models trained)

Coupled evaluation
  evaluate.sh
```

## Directory contents

| File | Purpose |
|------|---------|
| `uncoupled-atmos/train-config.yaml` | ACE2 atmosphere model pretraining config |
| `uncoupled-atmos/evaluator-config.yaml` | ACE2 evaluation (58,300 steps = ~40 years at 6h) |
| `uncoupled-ocean/train-config.yaml` | SamudraI ocean model pretraining config |
| `uncoupled-ocean/evaluator-config.yaml` | SamudraI evaluation (2,920 steps = ~40 years at 5-day) |
| `train-config-template.yaml` | SamudrACE stage 1 training config template |
| `finetune-config-template.yaml` | SamudrACE stage 2 training config template |
| `evaluator-config-ICx1.yaml` | SamudrACE evaluation from a single initial condition (year 311) |
| `train.sh` | Generates `train-config.yaml` and submits SamudrACE stage 1 training |
| `finetune.sh` | Generates `finetune-config.yaml` and submits SamudrACE stage 2 training |
| `evaluate.sh` | SamudrACE evaluation |
