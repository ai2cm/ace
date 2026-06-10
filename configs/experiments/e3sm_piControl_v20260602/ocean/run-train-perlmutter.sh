#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

COMMIT=$(git rev-parse --short HEAD)
export COMMIT

export FME_TRAIN_DIR=/pscratch/sd/e/elynnwu/fme-dataset
export FME_STATS_DIR=/pscratch/sd/e/elynnwu/fme-dataset/2026-06-02-E3SMv3-piControl-105yr-coupled-stats/ocean

TRAINING_FILE=${TRAINING_FILE:-training.txt}
CONFIG_FILE=${CONFIG_FILE:-config-train.yaml}
EXPERIMENTS_FILE=${EXPERIMENTS_FILE:-experiments.txt}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-sbatch-scripts/sbatch-train.sh}

if [[ ! -f "$TRAINING_FILE" ]]; then
  echo "Missing training file: $TRAINING_FILE"
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: $CONFIG_FILE"
  exit 1
fi

build_job_name() {
  local group="$1"
  local tag="$2"

  if [[ -n "$tag" ]]; then
    echo "${group}-${tag}"
  else
    echo "$group"
  fi
}

ensure_venv() {
  if [[ -n "${FME_VENV:-}" ]]; then
    return
  fi

  export FME_VENV=$(./make-venv.sh "$COMMIT" | tail -n 1)
  conda activate "$FME_VENV"
}

validate_config() {
  local config_path="$1"
  local override_args="$2"
  local validate_args=("$config_path" "--config_type" "train")

  if [[ -n "$override_args" ]]; then
    read -r -a override_array <<< "$override_args"
    validate_args+=("--override" "${override_array[@]}")
  fi

  python -m fme.ace.validate_config "${validate_args[@]}"
}

print_job() {
  echo
  echo "Perlmutter training job:"
  echo " - Job name: $JOB_NAME"
  echo " - WandB group: $WANDB_RUN_GROUP"
  echo " - Config: $CONFIG_FILE"
  echo " - Account/queue: ${ACCOUNT}/${QUEUE}"
  echo " - Constraint: $CONSTRAINT"
  echo " - Nodes: $NODES"
  echo " - GPUs per node: $GPUS_PER_NODE"
  echo " - CPUs per task: $CPUS_PER_TASK"
  echo " - Time: $TIME_LIMIT"
  echo " - Override: ${OVERRIDE_ARGS:-(none)}"
  echo " - Resume job: ${RESUME_JOB_ID:-(none)}"
}

TOTAL_JOBS=0
PROCESSED_JOBS=0
SKIPPED_JOBS=0

if [[ "$DRY_RUN" == "true" ]]; then
  echo "==========================================="
  echo "DRY RUN MODE - no Slurm jobs will be queued"
  echo "==========================================="
fi

while IFS= read -r TRAINING || [[ -n "$TRAINING" ]]; do
  [[ -z "$TRAINING" ]] && continue
  [[ "$TRAINING" =~ ^[[:space:]]*# ]] && continue
  [[ "$TRAINING" == group\|tag\|* ]] && continue

  TOTAL_JOBS=$((TOTAL_JOBS + 1))

  IFS="|" read -r \
    GROUP \
    TAG \
    STATUS \
    ACCOUNT \
    QUEUE \
    CONSTRAINT \
    NODES \
    GPUS_PER_NODE \
    CPUS_PER_TASK \
    TIME_LIMIT \
    OVERRIDE_ARGS \
    RESUME_JOB_ID \
    <<< "$TRAINING"

  if [[ "$STATUS" != "train" ]]; then
    SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
    continue
  fi

  PROCESSED_JOBS=$((PROCESSED_JOBS + 1))

  ACCOUNT=${ACCOUNT:-e3sm}
  QUEUE=${QUEUE:-regular}
  CONSTRAINT=${CONSTRAINT:-gpu&hbm80g}
  NODES=${NODES:-2}
  GPUS_PER_NODE=${GPUS_PER_NODE:-4}
  CPUS_PER_TASK=${CPUS_PER_TASK:-128}
  TIME_LIMIT=${TIME_LIMIT:-04:00:00}
  OVERRIDE_ARGS=${OVERRIDE_ARGS:-}
  RESUME_JOB_ID=${RESUME_JOB_ID:-}

  JOB_NAME=$(build_job_name "$GROUP" "$TAG")
  export WANDB_NAME="$JOB_NAME"
  export WANDB_RUN_GROUP="$GROUP"
  export FME_OVERRIDE_ARGS="$OVERRIDE_ARGS"

  if [[ -n "$RESUME_JOB_ID" ]]; then
    export RESUME_JOB_ID
  else
    unset RESUME_JOB_ID
  fi

  print_job

  if [[ "$DRY_RUN" == "true" ]]; then
    continue
  fi

  ensure_venv

  UUID=$(uuidgen)
  export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
  mkdir -p "$CONFIG_DIR" joblogs

  if [[ -z "${RESUME_JOB_ID:-}" ]]; then
    cp "$CONFIG_FILE" "$CONFIG_DIR/train-config.yaml"
  else
    cp "${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml" "$CONFIG_DIR/train-config.yaml"
  fi

  cp "$0" "$CONFIG_DIR/run-train-perlmutter.sh"
  cp "$TRAINING_FILE" "$CONFIG_DIR/training.txt"
  cp "$SBATCH_SCRIPT" "$CONFIG_DIR/sbatch-train.sh"
  cp sbatch-scripts/requeueable-train.sh "$CONFIG_DIR/requeueable-train.sh"
  cp make-venv.sh "$CONFIG_DIR/make-venv.sh"
  cp upload-to-beaker.sh "$CONFIG_DIR/upload-to-beaker.sh"
  printf "%s\n" "$OVERRIDE_ARGS" > "$CONFIG_DIR/override_args.txt"

  validate_config "$CONFIG_DIR/train-config.yaml" "$OVERRIDE_ARGS"

  SBATCH_OUTPUT=$(
    sbatch \
      --reservation=aigs_picontrol \
      -A "$ACCOUNT" \
      -q "$QUEUE" \
      -C "$CONSTRAINT" \
      -J "$JOB_NAME" \
      --nodes "$NODES" \
      --gpus-per-node "$GPUS_PER_NODE" \
      --cpus-per-task "$CPUS_PER_TASK" \
      -t "$TIME_LIMIT" \
      "$SBATCH_SCRIPT"
  )
  SLURM_JOB_ID=${SBATCH_OUTPUT##* }
  echo "$SBATCH_OUTPUT"

  {
    echo
    echo "${GROUP}|${TAG}|${SLURM_JOB_ID}|submitted|${ACCOUNT}|${QUEUE}|${CONSTRAINT}|${NODES}|${GPUS_PER_NODE}|${CPUS_PER_TASK}|${TIME_LIMIT}|${OVERRIDE_ARGS}"
  } >> "$EXPERIMENTS_FILE"
done < "$TRAINING_FILE"

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Summary:"
  echo " - Jobs in file: $TOTAL_JOBS"
  echo " - Would submit: $PROCESSED_JOBS"
  echo " - Would skip: $SKIPPED_JOBS"
fi
