#!/bin/bash
# Wrapper script for uncoupled training jobs
# Usage: uncoupled_train.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>] [--dry-run]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/uncoupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the 'train-config.yaml' to use."
  echo "  - --atmos_stats: Override atmosphere stats data path (optional)"
  echo "  - --ocean_stats: Override ocean stats data path (optional)"
  echo "  - --coupled_stats: Override with coupled stats dataset containing coupled_atmosphere and uncoupled_ocean subdirs (optional, mutually exclusive with --atmos_stats/--ocean_stats)"
  echo "  - --dry-run: Preview actions without launching jobs or committing changes"
  exit 1
fi

# Parse positional arguments
EXPERIMENT_DIR="$1"
CONFIG_SUBDIR="$2"
shift 2

# Parse dry-run flag first
parse_dry_run_flag "$@"

# Parse optional stats arguments
parse_stats_args "$@"
validate_stats_args

# Initialize script environment
init_script_environment

# Mode-specific configuration
FME_MODULE="fme.ace.train"
INPUT_FILE="training.txt"
CONFIG_FILENAME="train-config.yaml"

# Construct absolute paths for file operations
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Construct relative paths for gantry/python commands
CONFIG_PATH_REL="$EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

# Print dry-run header
print_dry_run_header

# Change to repo root so paths are valid
cd "$REPO_ROOT"

# Initialize counters for dry-run summary
TOTAL_JOBS=0
PROCESSED_JOBS=0
SKIPPED_JOBS=0
FIRST_JOB_PRINTED=false

while read TRAINING; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    GROUP=$(echo "$TRAINING" | cut -d"|" -f1)
    TAG=$(echo "$TRAINING" | cut -d"|" -f2)
    STATUS=$(echo "$TRAINING" | cut -d"|" -f3)
    PRIORITY=$(echo "$TRAINING" | cut -d"|" -f4)
    CLUSTER=$(echo "$TRAINING" | cut -d"|" -f5)
    N_GPUS=$(echo "$TRAINING" | cut -d"|" -f6)
    SHARED_MEM=$(echo "$TRAINING" | cut -d"|" -f7)
    RETRIES=$(echo "$TRAINING" | cut -d"|" -f8)
    WORKSPACE=$(echo "$TRAINING" | cut -d"|" -f9)
    OVERRIDE_ARGS=$(echo "$TRAINING" | cut -d"|" -f10)

    if [[ "$STATUS" != "train" ]]; then
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
        continue
    fi

    PROCESSED_JOBS=$((PROCESSED_JOBS + 1))

    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi

    JOB_GROUP="${GROUP}"
    JOB_NAME=$(build_job_name "$JOB_GROUP" "$TAG" "train")

    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # No checkpoints for uncoupled training
    CHECKPOINT_DATASET_ARGS=()

    # Print job info based on dry-run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ "$FIRST_JOB_PRINTED" == "false" ]]; then
            print_detailed_job_info
            FIRST_JOB_PRINTED=true
        else
            print_condensed_job_info "$JOB_NAME" "$CONFIG_PATH_REL" "$CLUSTER" "$N_GPUS" "$SHARED_MEM" "$PRIORITY"
        fi
    else
        echo
        echo "Launching uncoupled training job:"
        echo " - Job name: ${JOB_NAME}"
        echo " - Config: ${CONFIG_PATH_REL}"
        echo " - Priority: ${PRIORITY}"
        echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
        echo " - GPUs: ${N_GPUS}"
        echo " - Shared memory: ${SHARED_MEM}"
        echo " - Workspace: ${WORKSPACE}"
        echo " - Override: ${OVERRIDE_ARGS}"
    fi

    # Validate config (use relative path)
    python -m fme.ace.validate_config "$CONFIG_PATH_REL" --config_type train --override $OVERRIDE_ARGS

    # Commit config if changed (use absolute path)
    git_commit_and_push_with_dry_run "$CONFIG_PATH" "${JOB_NAME}" "$GIT_BRANCH"

    if [[ "$DRY_RUN" != "true" ]]; then
        echo
    fi

    # Run the job (use relative path for CONFIG_PATH)
    CONFIG_PATH="$CONFIG_PATH_REL" EXPERIMENT_ID=$(run_gantry_training_job_with_dry_run "Run uncoupled pretraining: ${JOB_GROUP}")

    # Append to experiments.txt
    append_to_experiments_file_with_dry_run "$EXPERIMENT_DIR" "$CONFIG_SUBDIR" "$JOB_GROUP" "$TAG" \
        "$EXPERIMENT_ID" "training" "best_inference_ckpt" "normal" "--not-preemptible" "$GIT_BRANCH"

done <"$INPUT_PATH"

# Print dry-run summary
print_dry_run_summary "$TOTAL_JOBS" "$PROCESSED_JOBS" "$SKIPPED_JOBS"
