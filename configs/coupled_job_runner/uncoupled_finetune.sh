#!/bin/bash
# Wrapper script for uncoupled fine-tuning jobs
# Usage: uncoupled_finetune.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/uncoupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the 'finetune-config-template.yaml' to use."
  echo "  - --atmos_stats: Override atmosphere stats data path (optional)"
  echo "  - --ocean_stats: Override ocean stats data path (optional)"
  echo "  - --coupled_stats: Override with coupled stats dataset containing coupled_atmosphere and uncoupled_ocean subdirs (optional, mutually exclusive with --atmos_stats/--ocean_stats)"
  exit 1
fi

# Parse positional arguments
EXPERIMENT_DIR="$1"
CONFIG_SUBDIR="$2"
shift 2

# Parse optional stats arguments
parse_stats_args "$@"
validate_stats_args
set_default_stats

# Set up paths
REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# Mode-specific configuration
FME_MODULE="fme.ace.train"
INPUT_FILE="finetuning.txt"
CONFIG_FILENAME="finetune-config.yaml"
TEMPLATE_CONFIG_FILENAME="finetune-config-template.yaml"

# Construct absolute paths for file operations
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"
TEMPLATE_CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$TEMPLATE_CONFIG_FILENAME"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Construct relative paths for gantry/python commands
CONFIG_PATH_REL="$EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

# Change to repo root so paths are valid
cd "$REPO_ROOT"

while read FINETUNING; do
    GROUP=$(echo "$FINETUNING" | cut -d"|" -f1)
    TAG=$(echo "$FINETUNING" | cut -d"|" -f2)
    WANDB_PROJECT=$(echo "$FINETUNING" | cut -d"|" -f3)
    WANDB_ID=$(echo "$FINETUNING" | cut -d"|" -f4)
    CKPT_TYPE=$(echo "$FINETUNING" | cut -d"|" -f5)
    STATUS=$(echo "$FINETUNING" | cut -d"|" -f6)
    PRIORITY=$(echo "$FINETUNING" | cut -d"|" -f7)
    CLUSTER=$(echo "$FINETUNING" | cut -d"|" -f8)
    N_GPUS=$(echo "$FINETUNING" | cut -d"|" -f9)
    SHARED_MEM=$(echo "$FINETUNING" | cut -d"|" -f10)
    RETRIES=$(echo "$FINETUNING" | cut -d"|" -f11)
    WORKSPACE=$(echo "$FINETUNING" | cut -d"|" -f12)
    OVERRIDE_ARGS=$(echo "$FINETUNING" | cut -d"|" -f13)
    EXISTING_RESULTS_DATASET=$(echo "$FINETUNING" | cut -d"|" -f14)

    if [[ "$STATUS" != "train" ]]; then
        continue
    fi

    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi

    JOB_GROUP="${GROUP}"
    if [[ -n "$TAG" ]]; then
        JOB_NAME="${JOB_GROUP}-${TAG}-train"
    else
        JOB_NAME="${JOB_GROUP}-train"
    fi

    # Get experiment dataset
    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXPER_ID=$(get_experiment_from_wandb "$WANDB_PROJECT" "$WANDB_ID")
        EXISTING_RESULTS_DATASET=$(get_beaker_dataset_from_experiment "$EXPER_ID")
    fi

    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # Create config from template
    bash "$SCRIPT_DIR/create_finetune_config.sh" \
        "$EXISTING_RESULTS_DATASET" \
        "$TEMPLATE_CONFIG_PATH" \
        "$CONFIG_PATH"

    # Build checkpoint dataset args for this job
    CHECKPOINT_DATASET_ARGS=(
        --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/$CKPT_TYPE.tar:/ckpt.tar"
    )

    echo
    echo "Launching uncoupled fine-tuning job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Config: ${CONFIG_PATH_REL}"
    echo " - Pretraining results dataset ID: ${EXISTING_RESULTS_DATASET}"
    echo " - Checkpoint type: ${CKPT_TYPE}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - Workspace: ${WORKSPACE}"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Override: ${OVERRIDE_ARGS}"

    # Validate config (use relative path)
    python -m fme.ace.validate_config "$CONFIG_PATH_REL" --config_type train --override $OVERRIDE_ARGS

    # Commit config if changed (use absolute path)
    git_commit_and_push "$CONFIG_PATH" "${JOB_NAME}" "$GIT_BRANCH"

    echo

    # Run the job (use relative path for CONFIG_PATH)
    CONFIG_PATH="$CONFIG_PATH_REL" EXPERIMENT_ID=$(run_gantry_training_job "Run uncoupled fine-tuning: ${JOB_GROUP}")

    # Append to experiments.txt
    { echo;
      echo "${JOB_GROUP}|${TAG}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git_commit_and_push "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "Update ${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "$GIT_BRANCH"

done <"$INPUT_PATH"
