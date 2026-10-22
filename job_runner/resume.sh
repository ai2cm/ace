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
  echo "Reads <experiment_dir>/<config_subdirectory>/resuming.txt, header:"
  echo "  $(input_header RESUMING_COLUMNS)"
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
    is_job_row "$RESUMING" || continue
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    parse_row "$RESUMING" RESUMING_COLUMNS
    if [[ -z $SHARED_MEM ]]; then
        SHARED_MEM=$(default_shared_mem "$CLUSTER" "$N_GPUS")
    fi

    if [[ "$STATUS" != "train" ]]; then
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
        continue
    fi

    PROCESSED_JOBS=$((PROCESSED_JOBS + 1))

    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi

    if [[ -z $MIN_RUNTIME ]]; then
        MIN_RUNTIME=$DEFAULT_MIN_RUNTIME
    fi

    JOB_GROUP="${GROUP}"
    JOB_NAME=$(build_job_name "$JOB_GROUP" "$TAG" "train")
    require_job_name_length "$JOB_NAME"

    # Get experiment dataset
    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXPER_ID=$(get_experiment_from_wandb "$WANDB_PROJECT" "$WANDB_ID")
        EXISTING_RESULTS_DATASET=$(get_beaker_dataset_from_experiment "$EXPER_ID")
    fi

    # Build cluster and stats args
    build_cluster_args "$CLUSTER" "$WORKSPACE"
    build_stats_dataset_args

    # Reset per iteration so one row's env does not leak into the next
    EXTRA_ENV_ARGS=()
    for ENV_KV in $EXTRA_ENV; do
        EXTRA_ENV_ARGS+=(--env "$ENV_KV")
    done

    # Blank config resumes the dumped config; else a repo config in the subdir
    if [[ -n $CONFIG ]]; then
        CONFIG_PATH="$EXPERIMENT_DIR/$CONFIG_SUBDIR/$CONFIG"
        if [[ ! -f "$CONFIG_PATH" ]]; then
            echo "Error: config $CONFIG_PATH not found" >&2
            exit 1
        fi
    else
        CONFIG_PATH="/existing-results/config.yaml"
    fi
    CONFIG_PATH_REL="$CONFIG_PATH"

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
        echo " - Extra env: ${EXTRA_ENV}"
        echo " - Override: ${OVERRIDE_ARGS}"
    fi

    TMPDIR_RESUME=$(mktemp -d)
    # Mount only what a resume reads, file by file: the whole results dataset
    # also holds best/EMA checkpoints and any .tmp left by a save that was
    # killed mid-write, which can multiply the download (and every retry's).
    CHECKPOINT_DATASET_ARGS=(
        --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/ckpt.tar:/existing-results/training_checkpoints/ckpt.tar"
        --dataset "$EXISTING_RESULTS_DATASET:wandb_run_id:/existing-results/wandb_run_id"
        --dataset "$EXISTING_RESULTS_DATASET:config.yaml:/existing-results/config.yaml"
    )
    RESULTS_LISTING=$(beaker dataset ls "$EXISTING_RESULTS_DATASET")
    if grep -q "metrics/metrics.jsonl" <<<"$RESULTS_LISTING"; then
        CHECKPOINT_DATASET_ARGS+=(
            --dataset "$EXISTING_RESULTS_DATASET:metrics/metrics.jsonl:/existing-results/metrics/metrics.jsonl"
        )
    fi
    # /ckpt.tar only when the resumed config loads it (a fine-tune's
    # stepper_training.parameter_init.checkpoint_path)
    beaker dataset fetch "$EXISTING_RESULTS_DATASET" --prefix config.yaml --output "$TMPDIR_RESUME" >/dev/null
    if grep -q "/ckpt.tar" "$TMPDIR_RESUME/config.yaml"; then
        CHECKPOINT_DATASET_ARGS+=(
            --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/ckpt.tar:/ckpt.tar"
        )
    fi
    rm -rf "$TMPDIR_RESUME"

    if [[ -n $EXISTING_RESULTS_ATMOS_DATASET ]]; then
        # this is unused but necessary for resuming coupled_train.sh runs
        CHECKPOINT_DATASET_ARGS+=(
            --dataset "$EXISTING_RESULTS_ATMOS_DATASET:training_checkpoints/ckpt.tar:/atmos_ckpt.tar"
        )
    fi

    if [[ -n $EXISTING_RESULTS_OCEAN_DATASET ]]; then
        # this is unused but necessary for resuming coupled_train.sh runs
        CHECKPOINT_DATASET_ARGS+=(
            --dataset "$EXISTING_RESULTS_OCEAN_DATASET:training_checkpoints/ckpt.tar:/ocean_ckpt.tar"
        )
    fi

    # Prepend resume override to OVERRIDE_ARGS
    OVERRIDE_ARGS="resume_results.existing_dir=/existing-results ${OVERRIDE_ARGS}"

    # Run the job using run_gantry_training_job
    EXPERIMENT_ID=$(MIN_RUNTIME="$MIN_RUNTIME" run_gantry_training_job_with_dry_run "Resume ${EXPERIMENT_DIR} pretraining: ${JOB_GROUP}")

    # Stop the loop if beaker did not return an experiment ID
    require_experiment_id "$EXPERIMENT_ID" "$JOB_NAME"

    # Append to experiments.txt
    append_to_experiments_file_with_dry_run "$EXPERIMENT_DIR" "$CONFIG_SUBDIR" "$JOB_GROUP" "$TAG" \
        "$EXPERIMENT_ID" "training" "best_inference_ckpt" "normal" "--min-runtime $DEFAULT_MIN_RUNTIME" "$GIT_BRANCH"

done <"$INPUT_PATH"

# Print submission and dry-run summaries
print_submission_summary
print_dry_run_summary "$TOTAL_JOBS" "$PROCESSED_JOBS" "$SKIPPED_JOBS"
