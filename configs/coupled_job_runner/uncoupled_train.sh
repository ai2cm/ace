#!/bin/bash
# Wrapper script for uncoupled training jobs
# Usage: uncoupled_train.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/uncoupled)"
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

# Mode-specific configuration
FME_MODULE="fme.ace.train"
INPUT_FILE="training.txt"
CONFIG_FILENAME="train-config.yaml"

# Construct full paths
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Change to repo root so paths are valid
cd "$REPO_ROOT"

while read TRAINING; do
    GROUP=$(echo "$TRAINING" | cut -d"|" -f1)
    STATUS=$(echo "$TRAINING" | cut -d"|" -f2)
    PRIORITY=$(echo "$TRAINING" | cut -d"|" -f3)
    CLUSTER=$(echo "$TRAINING" | cut -d"|" -f4)
    N_GPUS=$(echo "$TRAINING" | cut -d"|" -f5)
    SHARED_MEM=$(echo "$TRAINING" | cut -d"|" -f6)
    RETRIES=$(echo "$TRAINING" | cut -d"|" -f7)
    WORKSPACE=$(echo "$TRAINING" | cut -d"|" -f8)
    OVERRIDE_ARGS=$(echo "$TRAINING" | cut -d"|" -f9)

    if [[ "$STATUS" != "train" ]]; then
        continue
    fi

    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi

    JOB_GROUP="${GROUP}"
    JOB_NAME="${JOB_GROUP}-train"

    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # No checkpoints for uncoupled training
    CHECKPOINT_DATASET_ARGS=()

    echo
    echo "Launching uncoupled training job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Config: ${CONFIG_PATH}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Workspace: ${WORKSPACE}"
    echo " - Override: ${OVERRIDE_ARGS}"

    # Validate config
    python -m fme.ace.validate_config "$CONFIG_PATH" --config_type train $OVERRIDE_ARGS

    # Commit config if changed
    git_commit_and_push "$CONFIG_PATH" "${JOB_NAME}" "$GIT_BRANCH"

    echo

    # Run the job
    EXPERIMENT_ID=$(run_gantry_training_job "Run uncoupled pretraining: ${JOB_GROUP}")

    # Append to experiments.txt
    { echo;
      echo "${JOB_GROUP}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git_commit_and_push "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "Update ${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "$GIT_BRANCH"

done <"$INPUT_PATH"
