# 2026-09-16-fraw-gradient-map

`dL/dF_raw` at the pre-corrector frozen-precipitation output of two ACE2S
pre-corrector fine-tuning checkpoints, on a few `val_piC` batches:

- `fprec` (wandb `kj3ap00o`): pre-corrector optimization on `PRATEsfc`,
  `tendency_of_total_water_path_due_to_advection`,
  `total_frozen_precipitation_rate`.
- `prate-adv` (wandb `qdpcwkgr`): the same without
  `total_frozen_precipitation_rate`.

```
neg       : F_raw < 0                    force-positive zeroes it; no gradient to F_raw when F_final is scored
clipped   : F_pos > alpha P_pos          clip replaces it with alpha P_pos; no gradient to F_raw when F_final is scored
trainable : 0 <= F_raw <= alpha P_pos    F_final = F_raw
```

Variants (`trained`, `post_F`, `pre_F`) and the outputs are documented in
`fraw_gradient_map.py`. `config-<variant>.yaml` carries each run's
`stepper_training` verbatim, `corrector_loss` included. `run.sh <variant>`
launches on jupiter, one GPU.

Outputs in the job's `/results`: `fraw_gradient_map.nc`, `facts.json`.
