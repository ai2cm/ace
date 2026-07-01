# Perlmutter Job Runner

Centralized Perlmutter launch scripts for experiment directories that contain
`config-train.yaml`, `training.txt`, and `experiments.txt`.

## Training

```bash
./job_runner_pm/train.sh \
  configs/experiments/e3sm_piControl_v20260602/ocean \
  --stats /pscratch/sd/e/elynnwu/fme-dataset/2026-06-02-E3SMv3-piControl-105yr-coupled-stats/ocean
```

Useful options:

- `--dry-run`: print the jobs that would be submitted.
- `--train-dir <path>`: override `FME_TRAIN_DIR`.
- `--valid-dir <path>`: override `FME_VALID_DIR`.
- `--config-file <name>`: use a config other than `config-train.yaml`.
- `--training-file <name>`: use a training table other than `training.txt`.
- `--no-reservation`: submit without the default `aigs_picontrol` reservation.

The training table uses the same pipe-delimited format as the copied
`run-train-perlmutter.sh` scripts:

```text
group|tag|status|account|queue|constraint|nodes|gpus_per_node|cpus_per_task|time_limit|override_args|resume_job_id
```

Rows with `status=train` are submitted. All reusable shell helpers are staged to
`$PSCRATCH/fme-config/<uuid>` for each job so later local edits do not affect a
running Slurm job.

## Coupled Training From Pretrained Components

```bash
./job_runner_pm/coupled_train.sh \
  configs/experiments/my-coupled/pm \
  --stats /pscratch/sd/e/elynnwu/fme-dataset/coupled-stats
```

The experiment directory should contain a coupled template, defaulting to
`config-train-template.yaml`, and a `pretraining.txt` table:

```text
group|tag|status|ocean_config|ocean_ckpt|atmos_config|atmos_ckpt|account|queue|constraint|nodes|gpus_per_node|cpus_per_task|time_limit|override_args|resume_job_id
```

Rows with `status=train` generate `config-train.yaml` by merging the pretrained
ocean and atmosphere `stepper` sections into the template, preserving template
values when both configs specify the same field. Checkpoint paths are substituted
for `OCEAN_CKPT` and `ATMOS_CKPT` in the staged Slurm config.
