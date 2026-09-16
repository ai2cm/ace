# 2026-09-16-dtdalpha-map

The energy corrector's uniform temperature correction `dT`, the frozen clip's
share of it, and the sensitivity `d(dT)/d(alpha)` to the moisture rescale, at
the ACE2S fine-tuning checkpoint (wandb `ai2cm/ace2-cm4-atmos-noLSM/84vk67xi`,
`best_inference_ckpt`, epoch 2), on a few `val_piC` batches.

```
F_final   = min(F_pos, alpha P_pos)
dT_alpha  = dT(F_final) - dT(F_pos)
d(dT)/d(alpha) = L_f dt <P_pos [F_pos > alpha P_pos]> / <factor>
d(dT)/d(P_raw) = (L_f dt / <factor>) alpha a [P_raw > 0] ( [clip] - <P_pos [clip]> / <P_pos> )
```

- `dtdalpha_map.py`: loads the stepper, replays its correction sequence with
  detached outputs and `P_raw` a leaf, re-runs the total-energy step with the
  frozen field swapped (`F_pos`, `min(F_pos, s P_pos)` over a sweep in `s`),
  takes autograd of `dT` wrt `P_raw` and wrt `s`, checks both closed forms.
- `config.yaml`: checkpoint path, the run's `val_piC` loader window (smaller
  batch), `stepper_training` copied from the run config, the `s` sweep.
- `run.sh`: gantry launch, jupiter, high priority, `ai2/ace`, one GPU.
  `./run.sh [best_inference_ckpt|best_ckpt|ckpt]`.

Outputs in the job's `/results`: `dtdalpha_map.nc`, `facts.json`.
