#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

usage() {
  cat <<EOF
Usage: $0 <experiment_dir> --stats <path> [options]

Launch Perlmutter coupled training jobs from an experiment directory containing
a coupled config template and a pretraining table.

Required:
  <experiment_dir>             Directory containing the experiment config and inputs.
  --stats <path>               Stats directory to expose as FME_STATS_DIR.

Options:
  --train-dir <path>           Training dataset root. Defaults to /pscratch/sd/e/elynnwu/fme-dataset.
  --valid-dir <path>           Validation dataset root. Defaults to empty.
  --template-file <name>       Coupled template inside experiment_dir. Defaults to config-train-template.yaml.
  --config-file <name>         Generated config inside experiment_dir. Defaults to config-train.yaml.
  --pretraining-file <name>    Pretraining table inside experiment_dir. Defaults to pretraining.txt.
  --experiments-file <name>    Experiments log inside experiment_dir. Defaults to experiments.txt.
  --sbatch-script <path>       Override sbatch script. Defaults to job_runner_pm/sbatch-scripts/sbatch-train.sh.
  --reservation <name>         Slurm reservation. Defaults to aigs_picontrol.
  --no-reservation             Submit without a Slurm reservation.
  --dry-run                    Print jobs without creating envs or submitting to Slurm.
  -h, --help                   Show this help.

Pretraining table format:
  group|tag|status|ocean_config|ocean_ckpt|atmos_config|atmos_ckpt|account|queue|constraint|nodes|gpus_per_node|cpus_per_task|time_limit|override_args|resume_job_id
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
PRETRAINING_FILE="pretraining.txt"
CONFIG_FILE="config-train.yaml"
TEMPLATE_FILE="config-train-template.yaml"
EXPERIMENTS_FILE="experiments.txt"
SBATCH_SCRIPT="$SCRIPT_DIR/sbatch-scripts/sbatch-train.sh"
RESERVATION="aigs_picontrol"
USE_RESERVATION=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stats)
      FME_STATS_DIR="${2:?Missing value for --stats}"
      shift 2
      ;;
    --train-dir)
      FME_TRAIN_DIR="${2:?Missing value for --train-dir}"
      shift 2
      ;;
    --valid-dir)
      FME_VALID_DIR="${2:?Missing value for --valid-dir}"
      shift 2
      ;;
    --template-file)
      TEMPLATE_FILE="${2:?Missing value for --template-file}"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="${2:?Missing value for --config-file}"
      shift 2
      ;;
    --pretraining-file)
      PRETRAINING_FILE="${2:?Missing value for --pretraining-file}"
      shift 2
      ;;
    --experiments-file)
      EXPERIMENTS_FILE="${2:?Missing value for --experiments-file}"
      shift 2
      ;;
    --sbatch-script)
      SBATCH_SCRIPT="${2:?Missing value for --sbatch-script}"
      shift 2
      ;;
    --reservation)
      RESERVATION="${2:?Missing value for --reservation}"
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

PRETRAINING_PATH="$EXPERIMENT_DIR/$PRETRAINING_FILE"
CONFIG_PATH="$EXPERIMENT_DIR/$CONFIG_FILE"
TEMPLATE_PATH="$EXPERIMENT_DIR/$TEMPLATE_FILE"
EXPERIMENTS_PATH="$EXPERIMENT_DIR/$EXPERIMENTS_FILE"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
  echo "Missing experiment directory: $EXPERIMENT_DIR"
  exit 1
fi

if [[ ! -f "$PRETRAINING_PATH" ]]; then
  echo "Missing pretraining file: $PRETRAINING_PATH"
  exit 1
fi

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Missing template file: $TEMPLATE_PATH"
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
export FME_TRAIN_MODULE="fme.coupled.train"

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

resolve_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    echo "$path"
  else
    echo "$EXPERIMENT_DIR/$path"
  fi
}

validate_config() {
  local config_path="$1"
  local override_args="$2"
  local validate_args=("$config_path" "--config_type" "train")

  if [[ -n "$override_args" ]]; then
    read -r -a override_array <<< "$override_args"
    validate_args+=("--override" "${override_array[@]}")
  fi

  python -m fme.coupled.validate_config "${validate_args[@]}"
}

print_job() {
  echo
  echo "Perlmutter coupled training job:"
  echo " - Experiment dir: $EXPERIMENT_DIR"
  echo " - Job name: $JOB_NAME"
  echo " - WandB group: $WANDB_RUN_GROUP"
  echo " - Template: $TEMPLATE_PATH"
  echo " - Generated config: $CONFIG_PATH"
  echo " - Ocean config: $OCEAN_CONFIG"
  echo " - Ocean checkpoint: $OCEAN_CKPT"
  echo " - Atmosphere config: $ATMOS_CONFIG"
  echo " - Atmosphere checkpoint: $ATMOS_CKPT"
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

while IFS= read -r PRETRAINING || [[ -n "$PRETRAINING" ]]; do
  [[ -z "$PRETRAINING" ]] && continue
  [[ "$PRETRAINING" =~ ^[[:space:]]*# ]] && continue
  [[ "$PRETRAINING" == group\|tag\|* ]] && continue

  TOTAL_JOBS=$((TOTAL_JOBS + 1))

  IFS="|" read -r \
    GROUP \
    TAG \
    STATUS \
    OCEAN_CONFIG \
    OCEAN_CKPT \
    ATMOS_CONFIG \
    ATMOS_CKPT \
    ACCOUNT \
    QUEUE \
    CONSTRAINT \
    NODES \
    GPUS_PER_NODE \
    CPUS_PER_TASK \
    TIME_LIMIT \
    OVERRIDE_ARGS \
    RESUME_JOB_ID \
    <<< "$PRETRAINING"

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

  OCEAN_CONFIG=$(resolve_path "$OCEAN_CONFIG")
  ATMOS_CONFIG=$(resolve_path "$ATMOS_CONFIG")
  OCEAN_CKPT=$(resolve_path "$OCEAN_CKPT")
  ATMOS_CKPT=$(resolve_path "$ATMOS_CKPT")

  for REQUIRED_PATH in "$OCEAN_CONFIG" "$ATMOS_CONFIG" "$OCEAN_CKPT" "$ATMOS_CKPT"; do
    if [[ ! -e "$REQUIRED_PATH" ]]; then
      echo "Missing required path for job ${GROUP}: $REQUIRED_PATH"
      exit 1
    fi
  done

  JOB_NAME=$(build_job_name "$GROUP" "$TAG")
  export WANDB_NAME="$JOB_NAME"
  export WANDB_RUN_GROUP="$GROUP"
  export FME_OVERRIDE_ARGS="$OVERRIDE_ARGS"
  export OCEAN_CKPT
  export ATMOS_CKPT

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

  "$SCRIPT_DIR/create_coupled_train_config.py" \
    --atmos-config "$ATMOS_CONFIG" \
    --ocean-config "$OCEAN_CONFIG" \
    --template-config "$TEMPLATE_PATH" \
    --output-config "$CONFIG_PATH"

  UUID=$(uuidgen)
  export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
  mkdir -p "$CONFIG_DIR" joblogs

  if [[ -z "${RESUME_JOB_ID:-}" ]]; then
    cp "$CONFIG_PATH" "$CONFIG_DIR/train-config.yaml"
  else
    cp "${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml" "$CONFIG_DIR/train-config.yaml"
  fi

  cp "$SCRIPT_PATH" "$CONFIG_DIR/coupled_train.sh"
  cp "$PRETRAINING_PATH" "$CONFIG_DIR/pretraining.txt"
  cp "$TEMPLATE_PATH" "$CONFIG_DIR/$TEMPLATE_FILE"
  cp "$SBATCH_SCRIPT" "$CONFIG_DIR/sbatch-train.sh"
  cp "$SCRIPT_DIR/sbatch-scripts/requeueable-train.sh" "$CONFIG_DIR/requeueable-train.sh"
  cp "$SCRIPT_DIR/make-venv.sh" "$CONFIG_DIR/make-venv.sh"
  cp "$SCRIPT_DIR/upload-to-beaker.sh" "$CONFIG_DIR/upload-to-beaker.sh"
  cp "$SCRIPT_DIR/create_coupled_train_config.py" "$CONFIG_DIR/create_coupled_train_config.py"
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
done < "$PRETRAINING_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Summary:"
  echo " - Jobs in file: $TOTAL_JOBS"
  echo " - Would submit: $PROCESSED_JOBS"
  echo " - Would skip: $SKIPPED_JOBS"
fi
