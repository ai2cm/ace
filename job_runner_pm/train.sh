#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

usage() {
  cat <<EOF
Usage: $0 <experiment_dir> --stats <path> [options]

Launch Perlmutter training jobs from an experiment directory containing
config-train.yaml and training.txt.

Required:
  <experiment_dir>          Directory containing the experiment config and inputs.
  --stats <path>            Stats directory to expose as FME_STATS_DIR.

Options:
  --train-dir <path>        Training dataset root. Defaults to /pscratch/sd/e/elynnwu/fme-dataset.
  --valid-dir <path>        Validation dataset root. Defaults to empty.
  --config-file <name>      Config file inside experiment_dir. Defaults to config-train.yaml.
  --training-file <name>    Training table inside experiment_dir. Defaults to training.txt.
  --experiments-file <name> Experiments log inside experiment_dir. Defaults to experiments.txt.
  --sbatch-script <path>    Override sbatch script. Defaults to job_runner_pm/sbatch-scripts/sbatch-train.sh.
  --reservation <name>      Slurm reservation. Defaults to aigs_picontrol.
  --no-reservation          Submit without a Slurm reservation.
  --dry-run                 Print jobs without creating envs or submitting to Slurm.
  -h, --help                Show this help.
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 1
fi

EXPERIMENT_DIR=""
FME_STATS_DIR=""
FME_TRAIN_DIR="/pscratch/sd/e/elynnwu/fme-dataset"
FME_VALID_DIR=""
TRAINING_FILE="training.txt"
CONFIG_FILE="config-train.yaml"
EXPERIMENTS_FILE="experiments.txt"
SBATCH_SCRIPT="$SCRIPT_DIR/sbatch-scripts/sbatch-train.sh"
RESERVATION="aigs_picontrol"
USE_RESERVATION=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stats)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --stats"
        exit 1
      fi
      FME_STATS_DIR="$2"
      shift 2
      ;;
    --train-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --train-dir"
        exit 1
      fi
      FME_TRAIN_DIR="$2"
      shift 2
      ;;
    --valid-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --valid-dir"
        exit 1
      fi
      FME_VALID_DIR="$2"
      shift 2
      ;;
    --config-file)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config-file"
        exit 1
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    --training-file)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --training-file"
        exit 1
      fi
      TRAINING_FILE="$2"
      shift 2
      ;;
    --experiments-file)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --experiments-file"
        exit 1
      fi
      EXPERIMENTS_FILE="$2"
      shift 2
      ;;
    --sbatch-script)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --sbatch-script"
        exit 1
      fi
      SBATCH_SCRIPT="$2"
      shift 2
      ;;
    --reservation)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --reservation"
        exit 1
      fi
      RESERVATION="$2"
      USE_RESERVATION=true
      shift 2
      ;;
    --no-reservation)
      USE_RESERVATION=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [[ -n "$EXPERIMENT_DIR" ]]; then
        echo "Unexpected positional argument: $1"
        usage
        exit 1
      fi
      EXPERIMENT_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$EXPERIMENT_DIR" ]]; then
  echo "Missing experiment directory."
  usage
  exit 1
fi

if [[ -z "$FME_STATS_DIR" ]]; then
  echo "Missing required --stats <path>."
  usage
  exit 1
fi

if [[ "$EXPERIMENT_DIR" != /* ]]; then
  EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
fi

if [[ "$SBATCH_SCRIPT" != /* ]]; then
  SBATCH_SCRIPT="$REPO_ROOT/$SBATCH_SCRIPT"
fi

TRAINING_PATH="$EXPERIMENT_DIR/$TRAINING_FILE"
CONFIG_PATH="$EXPERIMENT_DIR/$CONFIG_FILE"
EXPERIMENTS_PATH="$EXPERIMENT_DIR/$EXPERIMENTS_FILE"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
  echo "Missing experiment directory: $EXPERIMENT_DIR"
  exit 1
fi

if [[ ! -f "$TRAINING_PATH" ]]; then
  echo "Missing training file: $TRAINING_PATH"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config file: $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT"
  exit 1
fi

COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
export COMMIT
export FME_TRAIN_DIR
export FME_VALID_DIR
export FME_STATS_DIR

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

  export FME_VENV=$("$SCRIPT_DIR/make-venv.sh" "$COMMIT" | tail -n 1)
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
  echo " - Experiment dir: $EXPERIMENT_DIR"
  echo " - Job name: $JOB_NAME"
  echo " - WandB group: $WANDB_RUN_GROUP"
  echo " - Config: $CONFIG_PATH"
  echo " - Training file: $TRAINING_PATH"
  echo " - Train dir: $FME_TRAIN_DIR"
  echo " - Valid dir: ${FME_VALID_DIR:-(empty)}"
  echo " - Stats dir: $FME_STATS_DIR"
  echo " - Account/queue: ${ACCOUNT}/${QUEUE}"
  echo " - Constraint: $CONSTRAINT"
  echo " - Nodes: $NODES"
  echo " - GPUs per node: $GPUS_PER_NODE"
  echo " - CPUs per task: $CPUS_PER_TASK"
  echo " - Time: $TIME_LIMIT"
  echo " - Reservation: $([[ "$USE_RESERVATION" == "true" ]] && echo "$RESERVATION" || echo "(none)")"
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

cd "$EXPERIMENT_DIR"

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
    cp "$CONFIG_PATH" "$CONFIG_DIR/train-config.yaml"
  else
    cp "${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml" "$CONFIG_DIR/train-config.yaml"
  fi

  cp "$0" "$CONFIG_DIR/train.sh"
  cp "$TRAINING_PATH" "$CONFIG_DIR/training.txt"
  cp "$SBATCH_SCRIPT" "$CONFIG_DIR/sbatch-train.sh"
  cp "$SCRIPT_DIR/sbatch-scripts/requeueable-train.sh" "$CONFIG_DIR/requeueable-train.sh"
  cp "$SCRIPT_DIR/make-venv.sh" "$CONFIG_DIR/make-venv.sh"
  cp "$SCRIPT_DIR/upload-to-beaker.sh" "$CONFIG_DIR/upload-to-beaker.sh"
  printf "%s\n" "$OVERRIDE_ARGS" > "$CONFIG_DIR/override_args.txt"

  validate_config "$CONFIG_DIR/train-config.yaml" "$OVERRIDE_ARGS"

  SBATCH_ARGS=()
  if [[ "$USE_RESERVATION" == "true" ]]; then
    SBATCH_ARGS+=(--reservation="$RESERVATION")
  fi

  SBATCH_OUTPUT=$(
    sbatch \
      "${SBATCH_ARGS[@]}" \
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
  } >> "$EXPERIMENTS_PATH"
done < "$TRAINING_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Summary:"
  echo " - Jobs in file: $TOTAL_JOBS"
  echo " - Would submit: $PROCESSED_JOBS"
  echo " - Would skip: $SKIPPED_JOBS"
fi
