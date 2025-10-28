#!/bin/bash
# Wrapper script for coupled training jobs
# Usage: coupled_train.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/coupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the 'train-config.yaml' to use."
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
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}

# Mode-specific configuration
FME_MODULE="fme.coupled.train"
INPUT_FILE="pretraining.txt"
CONFIG_FILENAME="train-config.yaml"
TEMPLATE_CONFIG_FILENAME="train-config-template.yaml"

# Construct absolute paths for file operations
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"
TEMPLATE_CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$TEMPLATE_CONFIG_FILENAME"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Construct relative paths for gantry/python commands
CONFIG_PATH_REL="$EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

print_stats_config

# Change to repo root so paths are valid
cd "$REPO_ROOT"

while read PRETRAINING; do
    GROUP=$(echo "$PRETRAINING" | cut -d"|" -f1)
    TAG=$(echo "$PRETRAINING" | cut -d"|" -f2)
    OCEAN_PROJECT=$(echo "$PRETRAINING" | cut -d"|" -f3)
    OCEAN_WANDB_ID=$(echo "$PRETRAINING" | cut -d"|" -f4)
    OCEAN_CKPT=$(echo "$PRETRAINING" | cut -d"|" -f5)
    ATMOS_PROJECT=$(echo "$PRETRAINING" | cut -d"|" -f6)
    ATMOS_WANDB_ID=$(echo "$PRETRAINING" | cut -d"|" -f7)
    ATMOS_CKPT=$(echo "$PRETRAINING" | cut -d"|" -f8)
    STATUS=$(echo "$PRETRAINING" | cut -d"|" -f9)
    PRIORITY=$(echo "$PRETRAINING" | cut -d"|" -f10)
    CLUSTER=$(echo "$PRETRAINING" | cut -d"|" -f11)
    N_GPUS=$(echo "$PRETRAINING" | cut -d"|" -f12)
    SHARED_MEM=$(echo "$PRETRAINING" | cut -d"|" -f13)
    RETRIES=$(echo "$PRETRAINING" | cut -d"|" -f14)
    WORKSPACE=$(echo "$PRETRAINING" | cut -d"|" -f15)
    OVERRIDE_ARGS=$(echo "$PRETRAINING" | cut -d"|" -f16)
    EXISTING_RESULTS_ATMOS_DATASET=$(echo "$PRETRAINING" | cut -d"|" -f17)
    EXISTING_RESULTS_OCEAN_DATASET=$(echo "$PRETRAINING" | cut -d"|" -f18)

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

    # Get experiment IDs and datasets
    if [[ -z "${EXISTING_RESULTS_ATMOS_DATASET}" ]]; then
        ATMOS_EXPER_ID=$(get_experiment_from_wandb "$ATMOS_PROJECT" "$ATMOS_WANDB_ID")
        EXISTING_RESULTS_ATMOS_DATASET=$(get_beaker_dataset_from_experiment "$ATMOS_EXPER_ID")
    fi

    if [[ -z "${EXISTING_RESULTS_OCEAN_DATASET}" ]]; then
        echo "Getting ocean experiment ID from wandb..."
        OCEAN_EXPER_ID=$(get_experiment_from_wandb "$OCEAN_PROJECT" "$OCEAN_WANDB_ID")
        EXISTING_RESULTS_OCEAN_DATASET=$(get_beaker_dataset_from_experiment "$OCEAN_EXPER_ID")
    fi
    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # Create config from template
    bash "$SCRIPT_DIR/create_coupled_train_config.sh" \
        "$EXISTING_RESULTS_ATMOS_DATASET" \
        "$EXISTING_RESULTS_OCEAN_DATASET" \
        "$TEMPLATE_CONFIG_PATH" \
        "$CONFIG_PATH"

    # Build checkpoint dataset args for this job
    CHECKPOINT_DATASET_ARGS=(
        --dataset "$EXISTING_RESULTS_ATMOS_DATASET:training_checkpoints/$ATMOS_CKPT.tar:/atmos_ckpt.tar"
        --dataset "$EXISTING_RESULTS_OCEAN_DATASET:training_checkpoints/$OCEAN_CKPT.tar:/ocean_ckpt.tar"
    )

    echo
    echo "Launching coupled training job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Config: ${CONFIG_PATH_REL}"
    echo " - Atmosphere training experiment ID: ${ATMOS_EXPER_ID}"
    echo " - Atmosphere results dataset ID: ${EXISTING_RESULTS_ATMOS_DATASET}"
    echo " - Atmosphere checkpoint type: ${ATMOS_CKPT}"
    echo " - Ocean training experiment ID: ${OCEAN_EXPER_ID}"
    echo " - Ocean results dataset ID: ${EXISTING_RESULTS_OCEAN_DATASET}"
    echo " - Ocean checkpoint type: ${OCEAN_CKPT}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Override: ${OVERRIDE_ARGS}"

    # Validate config (use relative path)
    python -m fme.coupled.validate_config "$CONFIG_PATH_REL" --config_type train $OVERRIDE_ARGS

    # Commit config if changed (use absolute path)
    git_commit_and_push "$CONFIG_PATH" "${JOB_NAME}" "$GIT_BRANCH"

    echo

    # Run the job (use relative path for CONFIG_PATH)
    CONFIG_PATH="$CONFIG_PATH_REL" EXPERIMENT_ID=$(run_gantry_training_job "Run coupled training from uncoupled pretraining: ${JOB_GROUP}")

    # Append to experiments.txt
    { echo;
      echo "${JOB_GROUP}|${TAG}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git_commit_and_push "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "Update ${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "$GIT_BRANCH"

done <"$INPUT_PATH"
