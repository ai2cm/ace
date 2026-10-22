# coupled_lin8_ppad_muon_alt: alternating coupled FT, 1°

Coupled fine-tune of the socorr lnmuon_fm lin8 stochastic ocean (wandb
`ace-samudra-cm4/3ly4s80y`, `best_inference_ckpt`) with the ACE2S atmosphere
(`ace2-cm4-atmos-noLSM/j0ng1tw3`). Both components train. Each batch updates
one of them, chosen 50/50 (`optimize_single_component_per_batch: true`). The
FTO of the same ocean (atmosphere frozen) is `../coupled_lin8_ppad_muon`, row
`socorrlnmuonfm_a2s-fto`.

## Recipe

`train-config-template.yaml` is `../coupled_lin8_ppad_muon/train-config-template.yaml`
(the FTO) with these changes:

- The atmosphere is unfrozen, with `loss_weight: 1.0`. Its step schedule comes
  from `../coupled/finetune-config-template.yaml` (1 to 80 steps, mostly 1 or 2,
  loss on the last step only).
- `optimize_single_component_per_batch: true`
- `optimization.max_grad_norm: 1.0`
- `ema_checkpoint_save_epochs: {step: 1}` saves every epoch's EMA checkpoint,
  so the evaluated epoch can be chosen afterwards.

There is one optimizer, so both components share one lr. Each row sets it.

## Memory (1 sample/GPU = global batch 8 on 8 GPUs)

| hardware | atmosphere checkpointing | peak reserved / GPU |
|---|---|---|
| 8x H100 | 0 | OOM in the first training steps (79 GiB) |
| 8x B200 | 0 | 111-112 GB |
| 4x B200 (2 samples/GPU) | 0 | ~200 GB estimated, over the 192 GB B200 |
| 4x B200 (2 samples/GPU) | 3 (every SFNO block) | see the experiment record |

The ocean already uses `checkpoint_strategy: all`. The atmosphere's lever is
`stepper.atmosphere.stepper.step.config.builder.config.checkpointing` (0 none,
1 encoder/decoder, 3 every block). It is a setting of the module, not stored
state, so it can be changed on a resume. Titan jobs should use 4 GPUs.

## Runs and how each was launched

All commands are run from the repo root on `exp/muon-ppad-stoch` with
`WANDB_USERNAME=elynnwu` and
`--coupled_stats jamesd/2026-07-15-om4-1pctco2-1deg-coupled-stats`.

1. **Fresh arms** (`pretraining.txt`, `job_runner/coupled_train.sh <this dir> .`).
   Rows `ftalt_lr7e-5` and `ftalt_lr1e-5`. The lr is set with
   `optimization.lr=` in the override column. Epoch 1 ran on 8x B200 with no
   atmosphere checkpointing. The first epoch is a linear warmup (0.01x to 1x over
   1460 batches), then the lr stays constant.

2. **Moving a running arm to 4 GPUs** (`resuming.txt`, `job_runner/resume.sh <this dir> .`):
   1. Stop the 8-GPU experiment, then rename it with a `-8xb200` suffix
      (`beaker experiment rename`). This frees its name: the job name becomes
      `WANDB_NAME`, and reusing the name keeps the wandb display name unchanged.
   2. Copy the stopped job's results dataset into `results_dataset`. Check that
      the dataset has `training_checkpoints/ckpt.tar` and `wandb_run_id`. A
      stopped job leaves its results dataset uncommitted, and gantry won't mount
      it (`BeakerError: uncommitted dataset`). Run `beaker dataset commit <id>`
      first. This also applies to step 3.
   3. Overrides: `resume_results.resume_wandb=true` (the default is a new wandb
      run) plus `...builder.config.checkpointing=3`. Leave `config` blank to
      resume from the dumped `/existing-results/config.yaml`.
   4. Mount only what is needed. Leave `results_dataset_ocean` / `_atmos`
      blank, and override
      `stepper_training.{ocean,atmosphere}.parameter_init.weights_path=null`:
      restoring the checkpoint overwrites the init weights anyway, and those
      mounts add ~16 GiB. `resume.sh` mounts only `ckpt.tar`, `wandb_run_id`,
      `config.yaml` and `metrics/metrics.jsonl` from `results_dataset`, plus
      `/ckpt.tar` when the resumed config loads it. Mounting the whole dataset
      instead pulled ~69 GiB per job, including spare checkpoints and a 9 GiB
      `.tmp`, and every preemption retry downloads it again.
   5. `resume.sh` does not validate. Validate by hand instead: fetch
      `config.yaml` from the results dataset, then run
      `python -m fme.coupled.validate_config config.yaml --config_type train --override <same overrides>`.
   6. Once it starts, the log should show `Resuming training from results` and
      the original wandb id.

3. **Branching a new schedule from an arm's checkpoint**
   (`finetuning.txt` + `finetune-config-template.yaml`,
   `job_runner/coupled_finetune.sh <this dir> .`). Row `ftalt_lr7e-5cos`:
   - Starts from `ckpt.tar` (raw weights, end of epoch 1) of the lr 7e-5 arm's
     8x job, pinned through `results_dataset`.
   - Creates a new wandb run with a fresh AdamW and a fresh EMA.
   - Uses a 146-batch warmup, then cosine 7e-5 to 7e-7 over the remaining
     7 epochs (`max_epochs: 7`).
   - `create_coupled_finetune_config.sh` merges only `.stepper` from the
     checkpoint, with the checkpoint winning on conflicts. Every
     `stepper_training` and `optimization` setting is therefore restated in the
     template. Set `parameter_init.weights_path: null` for each component next
     to `stepper_training.parameter_init.checkpoint_path`.
   - The finetune input file has no `extra_env` column, so this arm runs
     without `MALLOC_ARENA_MAX=4`.

Research record: `experiments/2026-10-21-socorr-lnmuon-fm-alt-coupled-ft/record.md`
in elynnwu's research repo.
