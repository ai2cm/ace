#!/bin/bash
# Wrapper script for resuming training jobs
# Usage: resume.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>] [--dry-run]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/coupled or experiments/2025-08-08-jamesd/uncoupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the 'resuming.txt' file."
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

# Determine FME module based on experiment directory
if [[ "$EXPERIMENT_DIR" =~ coupled ]]; then
    FME_MODULE="fme.coupled.train"
else
    FME_MODULE="fme.ace.train"
fi

# Input file
INPUT_FILE="resuming.txt"

# Construct full paths
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Print dry-run header
print_dry_run_header

# Change to repo root so paths are valid
cd "$REPO_ROOT"

# Initialize counters for dry-run summary
TOTAL_JOBS=0
PROCESSED_JOBS=0
SKIPPED_JOBS=0
FIRST_JOB_PRINTED=false

while read RESUMING; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    GROUP=$(echo "$RESUMING" | cut -d"|" -f1)
    TAG=$(echo "$RESUMING" | cut -d"|" -f2)
    WANDB_PROJECT=$(echo "$RESUMING" | cut -d"|" -f3)
    WANDB_ID=$(echo "$RESUMING" | cut -d"|" -f4)
    STATUS=$(echo "$RESUMING" | cut -d"|" -f5)
    PRIORITY=$(echo "$RESUMING" | cut -d"|" -f6)
    CLUSTER=$(echo "$RESUMING" | cut -d"|" -f7)
    N_GPUS=$(echo "$RESUMING" | cut -d"|" -f8)
    SHARED_MEM=$(echo "$RESUMING" | cut -d"|" -f9)
    RETRIES=$(echo "$RESUMING" | cut -d"|" -f10)
    WORKSPACE=$(echo "$RESUMING" | cut -d"|" -f11)
    OVERRIDE_ARGS=$(echo "$RESUMING" | cut -d"|" -f12)
    EXISTING_RESULTS_DATASET=$(echo "$RESUMING" | cut -d"|" -f13)

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

    # Get experiment dataset
    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXPER_ID=$(get_experiment_from_wandb "$WANDB_PROJECT" "$WANDB_ID")
        EXISTING_RESULTS_DATASET=$(get_beaker_dataset_from_experiment "$EXPER_ID")
    fi

    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # Set config path variable for print functions
    CONFIG_PATH_REL="/existing-results/config.yaml"

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
        echo "Resuming ${EXPERIMENT_DIR} training job:"
        echo " - Job name: ${JOB_NAME}"
        echo " - Resuming results dataset ID: ${EXISTING_RESULTS_DATASET}"
        echo " - Priority: ${PRIORITY}"
        echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
        echo " - Workspace: ${WORKSPACE}"
        echo " - GPUs: ${N_GPUS}"
        echo " - Shared memory: ${SHARED_MEM}"
        echo " - Override: ${OVERRIDE_ARGS}"
    fi

    # Build checkpoint dataset args for resume - includes entire results dir + checkpoint
    CHECKPOINT_DATASET_ARGS=(
        --dataset "$EXISTING_RESULTS_DATASET:/existing-results"
        --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/ckpt.tar:/ckpt.tar"
    )

    # Set config path to use the one from existing results
    CONFIG_PATH="/existing-results/config.yaml"

    # Prepend resume override to OVERRIDE_ARGS
    OVERRIDE_ARGS="resume_results.existing_dir=/existing-results ${OVERRIDE_ARGS}"

    # Run the job using run_gantry_training_job
    EXPERIMENT_ID=$(run_gantry_training_job_with_dry_run "Resume ${EXPERIMENT_DIR} pretraining: ${JOB_GROUP}")

    # Append to experiments.txt
    append_to_experiments_file_with_dry_run "$EXPERIMENT_DIR" "$CONFIG_SUBDIR" "$JOB_GROUP" "$TAG" \
        "$EXPERIMENT_ID" "training" "best_inference_ckpt" "normal" "--not-preemptible" "$GIT_BRANCH"

done <"$INPUT_PATH"

# Print dry-run summary
print_dry_run_summary "$TOTAL_JOBS" "$PROCESSED_JOBS" "$SKIPPED_JOBS"
