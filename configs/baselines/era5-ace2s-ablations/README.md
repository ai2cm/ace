# ACE2S paper ablation suite

40-epoch 1-step pretrains that perturb one knob of the stochastic ACE (ACE2S)
paper model's recipe, plus seed replicates of the unperturbed recipe. Each is
followed by a 10-epoch 3-step detached fine-tune once its pretrain finishes
(fine-tune configs are added here as they are launched).

The base is the paper model's pretrain config
(`configs/baselines/era5/ace-train-config-1-step-pretrain-daily-fg16-sr0p125-no-corr-mean.yaml`
on `experiment/no-corr-mean-outputs` at ace d90cd549, which trained
[gjsqlvsf](https://wandb.ai/ai2cm/ace/runs/gjsqlvsf)), with the evaluation
suite of its fine-tune config: 5-year inline inference every epoch (8 ICs,
3 members per IC, 5-day step means and ensemble metrics) and the 81-year
rollout (8 ICs, weight 0) every 5 epochs through epoch 40. Full checkpoints
are kept for epochs 36-40 (`checkpoint_save_epochs`) and EMA checkpoints
every 10 epochs (`ema_checkpoint_save_epochs`).

| Config | Knob |
|---|---|
| `ace2s-pretrain-rs{1,2,3}.yaml` | seed only (the paper pretrain is seed 0) |
| `ace2s-pretrain-deterministic.yaml` | `noise_embed_dim: 0`; MSE on one member with the ACE2 per-variable weights; inline inference with 1 member |
| `ace2s-pretrain-crps-only.yaml` | loss split 1.0 CRPS / 0.0 energy score (recipe: 0.9 / 0.1) |
| `ace2s-pretrain-no-bottleneck.yaml` | `filter_num_groups` and `spectral_ratio` dropped |
| `ace2s-pretrain-6hourly.yaml` | Troy's 6-hourly 2026-03-19 store and stats, 54 outputs (no `*_mean`); evaluation horizons in steps x4, 81-year rollout at epochs 10/20/30/40 |
| `ace2s-pretrain-4deg.yaml` | 4-degree: the regenerated `2026-09-08-era5-4deg-8layer-daily-1940-2025` store (`*_mean` fields plus PRMSL and the surface stresses in one store, from main's `scripts/data_process/configs/era5-4deg-8layer-1940-2025.yaml`); one GPU, batch size 8, loader parameters from the 4-degree daily v2 config; spectral bottleneck kept |
| `ace2s-pretrain-4deg-rs{1,2,3}.yaml` | the 4-degree arm with seeds 1-3 (a 4-member seed ensemble at 4 degrees, since one-GPU runs are cheap) |
| `ace2s-finetune-4deg.yaml` | the 4-degree arm's 10-epoch 3-step detached fine-tune (paper fine-tune recipe: `n_forward_steps: 3`, `use_gradient_accumulation: true`, warm start from the pretrain's `best_ckpt.tar` mounted at `/weights`); full and EMA checkpoints at epochs 5 and 10 |
| `ace2s-finetune-4deg-rs{1,2,3}.yaml` | the same fine-tune for the 4-degree seed replicates (seed N pretrain at `/weights`, `seed: N`) |
| `ace2s-finetune-rs1.yaml` | the 1-degree seed-1 replicate's 10-epoch 3-step detached fine-tune (paper fine-tune recipe, seed 1) |
| `ace2s-finetune-deterministic.yaml` | the deterministic arm's fine-tune: paper detached recipe with the arm's MSE loss (ACE2 weight table, single member) and 1-member inline inference |

Launch with `./run-train.sh [<filter> ...]` from this directory: workspace
`ai2/ace`, beaker priority `normal`, no `CM_PRIORITY` label, 8 GPUs on
jupiter (1 GPU for the 4-degree arm), `--min-runtime 8h`.
