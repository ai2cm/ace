#!/bin/bash
# Wrapper script for coupled training jobs
# Usage: coupled_train.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>] [--dry-run]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/coupled)"
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
FME_MODULE="fme.coupled.train"
INPUT_FILE="pretraining.txt"
CONFIG_FILENAME="train-config.yaml"
TEMPLATE_CONFIG_FILENAME="train-config-template.yaml"

# Construct absolute paths for file operations
FULL_EXPERIMENT_DIR="$REPO_ROOT/$EXPERIMENT_DIR"
CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"
INPUT_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$INPUT_FILE"

# Construct relative paths for gantry/python commands
CONFIG_PATH_REL="$EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

# Print dry-run header (includes stats config)
print_dry_run_header

# Print stats config if not in dry-run mode
if [[ "$DRY_RUN" != "true" ]]; then
    print_stats_config
fi

# Change to repo root so paths are valid
cd "$REPO_ROOT"

# Initialize counters for dry-run summary
TOTAL_JOBS=0
PROCESSED_JOBS=0
SKIPPED_JOBS=0
FIRST_JOB_PRINTED=false

while read PRETRAINING; do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
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

    if [[ "$CLUSTER" == "titan" ]]; then
        TEMPLATE_CONFIG_FILENAME="train-config-template.yaml"
    else
        TEMPLATE_CONFIG_FILENAME="train-config-template.yaml"
    fi
    TEMPLATE_CONFIG_PATH="$FULL_EXPERIMENT_DIR/$CONFIG_SUBDIR/$TEMPLATE_CONFIG_FILENAME"

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

    if [[ -z "$EXISTING_RESULTS_ATMOS_DATASET" ]]; then
        ATMOS_EXPER_ID=$(get_experiment_from_wandb "$ATMOS_PROJECT" "$ATMOS_WANDB_ID")
        EXISTING_RESULTS_ATMOS_DATASET=$(get_beaker_dataset_from_experiment "$ATMOS_EXPER_ID")
    else
        ATMOS_EXPER_ID="Not-used"
    fi
    if [[ -z "$EXISTING_RESULTS_OCEAN_DATASET" ]]; then
        OCEAN_EXPER_ID=$(get_experiment_from_wandb "$OCEAN_PROJECT" "$OCEAN_WANDB_ID")
        EXISTING_RESULTS_OCEAN_DATASET=$(get_beaker_dataset_from_experiment "$OCEAN_EXPER_ID")
    else
        OCEAN_EXPER_ID="Not-used"
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
    fi

    # Validate config (use relative path)
    python -m fme.coupled.validate_config "$CONFIG_PATH_REL" --config_type train --override $OVERRIDE_ARGS

    # Commit config if changed (use absolute path)
    git_commit_and_push_with_dry_run "$CONFIG_PATH" "${JOB_NAME}" "$GIT_BRANCH"

    if [[ "$DRY_RUN" != "true" ]]; then
        echo
    fi

    # Run the job (use relative path for CONFIG_PATH)
    CONFIG_PATH="$CONFIG_PATH_REL" EXPERIMENT_ID=$(run_gantry_training_job_with_dry_run "Run coupled training from uncoupled pretraining: ${JOB_GROUP}")

    # Append to experiments.txt
    append_to_experiments_file_with_dry_run "$EXPERIMENT_DIR" "$CONFIG_SUBDIR" "$JOB_GROUP" "$TAG" \
        "$EXPERIMENT_ID" "training" "best_inference_ckpt" "normal" "--not-preemptible" "$GIT_BRANCH"

done <"$INPUT_PATH"

# Print dry-run summary
print_dry_run_summary "$TOTAL_JOBS" "$PROCESSED_JOBS" "$SKIPPED_JOBS"
