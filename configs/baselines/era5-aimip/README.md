# ERA5 AIMIP Baseline

This directory contains scripts and configurations for training and running an ACE2 model on ERA5
data for the AIMIP evaluation protocol. This configuration is referred to as **ACE2.1-ERA5**. The
model predicts pressure-level diagnostic variables over 1978–2024 with multiple initial conditions
and SST perturbation scenarios.

## Workflow

The full experiment proceeds in five stages. Checkpoint IDs embedded in the scripts must be updated
manually after evaluating each stage.

### 1. Train

```bash
bash run-ace-train.sh
```

Trains 4 random-seed ensemble members (RS0–RS3) on ERA5 1979–2008, with validation on 2009–2014.
Config: `ace-train-config.yaml`.

### 2. Evaluate training seeds

```bash
bash run-ace-evaluator-seed-selection.sh
bash run-ace-evaluator-seed-selection-single.sh
```

Evaluate all 4 trained checkpoints to select the best seed for fine-tuning.

- `run-ace-evaluator-seed-selection.sh` — 7x 5-year evaluations (starting in 1980, 1985, 1990,
  1995, 2000, 2005, 2010). Config: `ace-evaluator-seed-selection-config.yaml`.
- `run-ace-evaluator-seed-selection-single.sh` — single continuous 36-year run (1978-10-01 to
  2014-12-31). Config: `ace-evaluator-seed-selection-single-config.yaml`.

The best seed is chosen based on comparing the time-mean climate and trend skill, both in
the 7x 5-year and 36-year evaluations; this is somewhat subjective. The chosen seed is used
in `run-ace-fine-tune-decoder-pressure-levels.sh`.

### 3. Fine-tune

```bash
bash run-ace-fine-tune-decoder-pressure-levels.sh
```

Freezes the best trained checkpoint and trains a secondary MLP decoder for 65 pressure-level
diagnostic variables (TMP, Q, UGRD, VGRD, h at 13 pressure levels plus near-surface fields) across
4 new random seeds. Config: `ace-fine-tune-pressure-level-separate-decoder-config.yaml`.

### 4. Evaluate fine-tuned seeds

Re-run both evaluator scripts from step 2. The scripts already include checkpoint IDs for both
the trained and fine-tuned ensemble members, enabling direct comparison. After evaluating seeds
similarly as before (though there is little variability due frozen prognostic state), the best
checkpoint ID is used in `run-ace-inference.sh`.

### 5. Run inference

```bash
bash run-ace-inference.sh
```

Runs 15 parallel 46-year simulations (1978-10-01 to 2024-12-31) using the best fine-tuned
checkpoint:

- 5 initial conditions (IC1–IC5) from the AIMIP IC dataset
- 3 SST scenarios: baseline, +2 K, +4 K

Configs: `ace-aimip-inference-config.yaml`, `ace-aimip-inference-p2k-config.yaml`,
`ace-aimip-inference-p4k-config.yaml`.
