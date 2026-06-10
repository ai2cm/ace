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
