# 2026-09-15-dldpraw-gradient-map

`dL_P/dP_raw` at the pre-corrector precipitation output of the ACE2S one-step
pretraining checkpoint (wandb `ai2cm/ace2-cm4-atmos-noLSM/jbg7a0z4`,
`best_ckpt`), on a few `val_piC` batches.

```
P_final = alpha * max(P_raw, 0),   alpha = (<E> - <dTWP/dt>) / <P_pos>
g[variant] = dL_variant / dP_raw
```

Variants and the closed-form check are documented in `gradient_map.py`.

- `gradient_map.py`: loads the stepper, wraps its corrector to keep the raw and
  corrected `PRATEsfc` on the graph, runs `TrainStepper.train_on_batch` with a
  `NullOptimization`, takes `autograd.grad` of each loss variant wrt `P_raw`.
- `config.yaml`: checkpoint path, loader (the run's `validation[2].loader`
  window, smaller batch), `stepper_training` copied from the run config.
- `run.sh`: gantry launch, jupiter, high priority, `ai2/ace`, one GPU.

Outputs in the job's `/results`: `gradient_map.nc`, `facts.json`.
