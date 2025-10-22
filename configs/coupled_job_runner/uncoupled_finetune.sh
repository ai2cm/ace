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

# Construct full paths
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"
TEMPLATE_CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$TEMPLATE_CONFIG_FILENAME"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Change to repo root so paths are valid
cd "$REPO_ROOT"

while read FINETUNING; do
    GROUP=$(echo "$FINETUNING" | cut -d"|" -f1)
    WANDB_PROJECT=$(echo "$FINETUNING" | cut -d"|" -f2)
    WANDB_ID=$(echo "$FINETUNING" | cut -d"|" -f3)
    CKPT_TYPE=$(echo "$FINETUNING" | cut -d"|" -f4)
    STATUS=$(echo "$FINETUNING" | cut -d"|" -f5)
    PRIORITY=$(echo "$FINETUNING" | cut -d"|" -f6)
    CLUSTER=$(echo "$FINETUNING" | cut -d"|" -f7)
    N_GPUS=$(echo "$FINETUNING" | cut -d"|" -f8)
    SHARED_MEM=$(echo "$FINETUNING" | cut -d"|" -f9)
    RETRIES=$(echo "$FINETUNING" | cut -d"|" -f10)
    WORKSPACE=$(echo "$FINETUNING" | cut -d"|" -f11)
    OVERRIDE_ARGS=$(echo "$FINETUNING" | cut -d"|" -f12)
    EXISTING_RESULTS_DATASET=$(echo "$FINETUNING" | cut -d"|" -f13)

    if [[ "$STATUS" != "train" ]]; then
        continue
    fi

    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi

    JOB_GROUP="${GROUP}"
    JOB_NAME="${JOB_GROUP}-train"

    # Get experiment dataset
    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXPER_ID=$(get_experiment_from_wandb "$WANDB_PROJECT" "$WANDB_ID")
        EXISTING_RESULTS_DATASET=$(get_beaker_dataset_from_experiment "$EXPER_ID")
    fi

    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # Create config from template
    bash "$FULL_EXPERIMENT_DIR/create_finetune_config.sh" \
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
    echo " - Config: ${CONFIG_PATH}"
    echo " - Pretraining results dataset ID: ${EXISTING_RESULTS_DATASET}"
    echo " - Checkpoint type: ${CKPT_TYPE}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - Workspace: ${WORKSPACE}"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Override: ${OVERRIDE_ARGS}"

    # Validate config
    python -m fme.ace.validate_config "$CONFIG_PATH" --config_type train $OVERRIDE_ARGS

    # Commit config if changed
    git_commit_and_push "$CONFIG_PATH" "${JOB_NAME}" "$GIT_BRANCH"

    echo

    # Run the job
    EXPERIMENT_ID=$(run_gantry_training_job "Run uncoupled fine-tuning: ${JOB_GROUP}")

    # Append to experiments.txt
    { echo;
      echo "${JOB_GROUP}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git_commit_and_push "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "Update ${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "$GIT_BRANCH"

done <"$INPUT_PATH"
