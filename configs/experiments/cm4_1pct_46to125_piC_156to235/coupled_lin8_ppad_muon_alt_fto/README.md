# coupled_lin8_ppad_muon_alt_fto: alternating coupled FT after the FTO, 1°

FTO-FTalt arm for comparison with the direct FTalt arms in
`../coupled_lin8_ppad_muon_alt`. It starts from the full coupled FTO checkpoint
(`../coupled_lin8_ppad_muon` row `socorrlnmuonfm_a2s-fto`, wandb
`ace-samudra-coupled-cm4/l0fc6hg8`, `best_inference_ckpt`) and runs 3 epochs at
jamesd's `coupled_clip-oppfm_a2sLR-ft_fto` schedule: AdamW lr 1e-5, LinearLR
0.01x to 1x over epoch 1 (1460 batches), then constant.

`finetune-config-template.yaml` is `../coupled_lin8_ppad_muon_alt/train-config-template.yaml`
with `stepper_training.parameter_init.checkpoint_path: /ckpt.tar` (component
`weights_path: null`), `max_epochs: 3`, and atmosphere activation checkpointing 3
for 4x B200 (see `../coupled_lin8_ppad_muon_alt/README.md`, Memory). Same lr and
schedule as that dir's `ftalt_lr1e-5` row, so the two differ only in init and
length.

Launch from the repo root on `exp/muon-ppad-stoch`:

```bash
WANDB_USERNAME=elynnwu bash job_runner/coupled_finetune.sh \
  configs/experiments/cm4_1pct_46to125_piC_156to235/coupled_lin8_ppad_muon_alt_fto . \
  --coupled_stats jamesd/2026-07-15-om4-1pctco2-1deg-coupled-stats
```

Research record: `experiments/2026-10-23-socorr-lnmuon-fm-fto-ftalt/record.md`
in elynnwu's research repo.
