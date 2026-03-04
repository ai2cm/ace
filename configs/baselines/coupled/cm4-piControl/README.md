# Coupled training and evaluation for CM4-piControl

Self-contained baseline configs and scripts for the SamudrACE coupled
atmosphere-ocean training pipeline on GFDL CM4 piControl data.

## Pipeline overview

The pipeline trains uncoupled atmosphere and ocean models independently, then
couples them via freeze-then-optimize (FTO) training, with an optional
joint fine-tuning stage. Each step produces a Beaker dataset whose ID is
plugged into the next script's `EXISTING_RESULTS_DATASET` variable.

```
Step 1: Uncoupled training (parallel)
  uncoupled-atmos/run-ace-train.sh   -> atmos checkpoint
  uncoupled-ocean/run-ace-train.sh   -> ocean checkpoint

Step 2 (optional): Uncoupled evaluation
  uncoupled-atmos/run-ace-evaluator.sh
  uncoupled-ocean/run-ace-evaluator.sh

Step 3: Coupled training (freeze-then-optimize)
  train.sh  -> coupled checkpoint (atmos frozen, ocean fine-tuned)

Step 4 (optional): Coupled fine-tuning
  finetune.sh  -> refined coupled checkpoint (both models trained)

Step 5: Coupled evaluation
  evaluate.sh
```

## Directory contents

| File | Purpose |
|------|---------|
| `uncoupled-atmos/ace-train-config.yaml` | SFNO atmosphere model: architecture, variables, loss weights |
| `uncoupled-atmos/ace-evaluator-config.yaml` | Atmosphere evaluation (58,300 steps = ~40 years at 6h) |
| `uncoupled-ocean/ace-train-config.yaml` | Samudra ocean model: architecture, variables, correctors |
| `uncoupled-ocean/ace-evaluator-config.yaml` | Ocean evaluation (2,920 steps = ~40 years at 5-day) |
| `train-config-template.yaml` | Coupled FTO training: data loaders, optimization, coupled stepper skeleton (atmos frozen, ocean trainable) |
| `finetune-config-template.yaml` | Coupled fine-tuning: lower LR, cosine annealing, both models trainable, loads from coupled checkpoint |
| `evaluator-config-ICx1.yaml` | Coupled evaluation from a single initial condition (year 311) |
| `train.sh` | Generates `coupled-train-config.yaml` and submits coupled training |
| `finetune.sh` | Generates `coupled-finetune-config.yaml` and submits fine-tuning |
| `evaluate.sh` | Submits coupled evaluation |

## How configs are generated

The coupled training configs are too large to maintain by hand since they
embed the full stepper definitions for both atmosphere and ocean models.
Instead, `train.sh` and `finetune.sh` generate them automatically:

1. Copy `uncoupled-atmos/ace-train-config.yaml` and
   `uncoupled-ocean/ace-train-config.yaml` to temp files
2. Remap stats paths (`statsdata` -> `atmos_stats` / `ocean_stats`)
3. Strip training-specific fields (`loss`, `parameter_init`, etc.)
4. Extract `sea_ice_fraction_name` from the ocean corrector config
5. Merge both steppers into the template (template values win on conflict)
6. Set `ocean_fraction_prediction.sea_ice_fraction_name`

This requires **yq >= 4** (`brew install yq` or `pip install yq`).

## How to use

1. **Train uncoupled models** -- run `uncoupled-atmos/run-ace-train.sh` and
   `uncoupled-ocean/run-ace-train.sh`. When complete, find the Beaker result
   dataset ID for each job.

2. **Update `train.sh`** -- set `EXISTING_RESULTS_ATMOS_DATASET` and
   `EXISTING_RESULTS_OCEAN_DATASET` to the dataset IDs from step 1.

3. **Run coupled training** -- run `train.sh`. This generates
   `coupled-train-config.yaml` and submits the job.

4. **(Optional) Fine-tune** -- set `EXISTING_RESULTS_DATASET` in
   `finetune.sh` to the dataset ID from coupled training, then run it.

5. **Evaluate** -- set `EXISTING_RESULTS_DATASET` in `evaluate.sh` to the
   dataset ID from coupled training (or fine-tuning), then run it.

6. **(Optional) Evaluate uncoupled models** -- set `EXISTING_RESULTS_DATASET`
   in the uncoupled evaluator scripts and run them.

## Key model details

- **Atmosphere**: SphericalFourierNeuralOperatorNet (SFNO), embed_dim=384,
  8 layers, 6h timestep, 8-level vertical discretization
- **Ocean**: Samudra CNN, ch_width=[200,250,300,400], 5-day timestep,
  19 depth levels for temperature/salinity/velocity
- **Coupled FTO**: 20 epochs, 4 coupled steps, atmosphere frozen, ocean
  trained with MSE loss
- **Coupled fine-tuning**: 20 epochs, lr=1e-5 with cosine annealing,
  both models trained, loads from coupled checkpoint
- **Data**: CM4 piControl 200-year simulation, train years 151-306,
  validation years 306-311, evaluation from year 311
