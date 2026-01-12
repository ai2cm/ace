#!/bin/bash
# Shared library for coupled/uncoupled training job scripts
# Provides common functions for argument parsing, dataset mounting, and gantry job submission

set -e

# Parse stats-related command line arguments
# Sets ATMOS_STATS_DATA, OCEAN_STATS_DATA, and COUPLED_STATS_DATA variables
# Usage: parse_stats_args "$@"
parse_stats_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --atmos_stats)
                ATMOS_STATS_DATA="$2"
                shift 2
                ;;
            --ocean_stats)
                OCEAN_STATS_DATA="$2"
                shift 2
                ;;
            --coupled_stats)
                COUPLED_STATS_DATA="$2"
                shift 2
                ;;
            *)
                # Unknown option - let the caller handle it
                shift
                ;;
        esac
    done
}

# Validate that --coupled_stats is not used with --atmos_stats or --ocean_stats
# Exits with error if validation fails
validate_stats_args() {
    if [[ -n "$COUPLED_STATS_DATA" ]] && [[ -n "$ATMOS_STATS_DATA" || -n "$OCEAN_STATS_DATA" ]]; then
        echo "Error: --coupled_stats cannot be used together with --atmos_stats or --ocean_stats"
        exit 1
    fi
    if [[ -z "$COUPLED_STATS_DATA" ]] && [[ -z "$ATMOS_STATS_DATA" ]] && [[ -z "$OCEAN_STATS_DATA" ]]; then
        echo "Error: stats must be provided"
        exit 1
    fi
}

# Build CLUSTER_ARGS array based on cluster and workspace parameters
# Args: $1 = CLUSTER, $2 = WORKSPACE (optional)
# Sets global CLUSTER_ARGS array and WORKSPACE variable
build_cluster_args() {
    local CLUSTER="$1"
    local WORKSPACE="${2:-}"

    if [[ -z "$WORKSPACE" ]]; then
        WORKSPACE=ai2/ace
    fi
    if [[ "$CLUSTER" == "a100+h100" ]]; then
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster ceres
            --cluster jupiter
            --cluster saturn
        )
    elif [[ -z "$CLUSTER" ]]; then
        # Default: h100
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster ceres
            --cluster jupiter
        )
    else
        # Specific cluster provided
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster "$CLUSTER"
        )
    fi

    # Export WORKSPACE so it's available to caller
    export WORKSPACE
}

# Build STATS_DATASET_ARGS array based on stats configuration
# Handles both coupled and separate stats datasets
# Sets global STATS_DATASET_ARGS array
build_stats_dataset_args() {
    if [[ -n "$COUPLED_STATS_DATA" ]]; then
        STATS_DATASET_ARGS=(
            --dataset "$COUPLED_STATS_DATA:coupled_atmosphere:/atmos_stats"
            --dataset "$COUPLED_STATS_DATA:ocean:/ocean_stats"
        )
    else
        STATS_DATASET_ARGS=()
        if [[ -n $ATMOS_STATS_DATA ]]; then
            STATS_DATASET_ARGS+=(--dataset "$ATMOS_STATS_DATA:/atmos_stats")
        fi
        if [[ -n $OCEAN_STATS_DATA ]]; then
            STATS_DATASET_ARGS+=(--dataset "$OCEAN_STATS_DATA:/ocean_stats")
        fi
    fi
}

# Get beaker dataset ID from experiment ID
# Args: $1 = EXPERIMENT_ID
# Outputs: dataset ID to stdout
get_beaker_dataset_from_experiment() {
    local EXPER_ID="$1"
    beaker experiment get "$EXPER_ID" --format json |
        jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
}

# Convert wandb project and run ID to beaker experiment ID
# Args: $1 = PROJECT, $2 = WANDB_ID
# Outputs: experiment ID to stdout
get_experiment_from_wandb() {
    local PROJECT="$1"
    local WANDB_ID="$2"
    local REPO_ROOT=$(git rev-parse --show-toplevel)

    python "$REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py" \
        --project "$PROJECT" --wandb_id "$WANDB_ID"
}

# Commit and push a file to git
# Args: $1 = FILE_PATH, $2 = COMMIT_MESSAGE, $3 = GIT_BRANCH
git_commit_and_push() {
    local FILE_PATH="$1"
    local COMMIT_MESSAGE="$2"
    local GIT_BRANCH="$3"

    if git status --porcelain "$FILE_PATH" | grep -q .; then
        git add "$FILE_PATH"
        git commit -m"$COMMIT_MESSAGE"
        git push origin "$GIT_BRANCH"
    fi
}

# Run a gantry training job with common parameters
# This function expects several variables to be set by the caller:
#   Required:
#     JOB_NAME, JOB_GROUP, PRIORITY, RETRIES, N_GPUS, SHARED_MEM
#     FME_MODULE, CONFIG_PATH (relative to repo root), BEAKER_USERNAME
#     CLUSTER_ARGS (array), STATS_DATASET_ARGS (array)
#   Optional:
#     CHECKPOINT_DATASET_ARGS (array) - additional dataset mounts
#     OVERRIDE_ARGS - config override arguments
#     PREEMPTIBLE (default: --preemptible)
#   Note: CONFIG_PATH must be relative to repo root for gantry run
# Returns: Experiment ID
run_gantry_training_job() {
    local REPO_ROOT=$(git rev-parse --show-toplevel)
    local DESCRIPTION="${1:-Training job}"
    local PREEMPTIBLE="${PREEMPTIBLE:---preemptible}"

    # Build override string
    local OVERRIDE=""
    if [[ -n "${OVERRIDE_ARGS:-}" ]]; then
        OVERRIDE="--override ${OVERRIDE_ARGS}"
    fi

    # Initialize checkpoint args if not set
    if [[ -z "${CHECKPOINT_DATASET_ARGS+x}" ]]; then
        CHECKPOINT_DATASET_ARGS=()
    fi

    local EXPERIMENT_ID=$(
        gantry run \
            --name "$JOB_NAME" \
            --description "$DESCRIPTION" \
            --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
            --priority "$PRIORITY" \
            $PREEMPTIBLE \
            --retries "$RETRIES" \
            "${CLUSTER_ARGS[@]}" \
            --weka climate-default:/climate-default \
            --env WANDB_USERNAME="$BEAKER_USERNAME" \
            --env WANDB_NAME="$JOB_NAME" \
            --env WANDB_JOB_TYPE=training \
            --env WANDB_RUN_GROUP="$JOB_GROUP" \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env NCCL_DEBUG=WARN \
            --env NCCL_DEBUG_FILE=/results/nccl_debug.log \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            "${STATS_DATASET_ARGS[@]}" \
            "${CHECKPOINT_DATASET_ARGS[@]}" \
            --gpus "$N_GPUS" \
            --shared-memory "$SHARED_MEM" \
            --budget ai2/climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node "$N_GPUS" -m $FME_MODULE "$CONFIG_PATH" $OVERRIDE |
            tee /dev/tty |
            sed -n 's|.*https://beaker\.org/ex/\([A-Z0-9]*\).*|\1|p'
    )

    echo "$EXPERIMENT_ID"
}

# Print stats configuration for debugging
print_stats_config() {
    echo
    if [[ -n "${COUPLED_STATS_DATA:-}" ]]; then
        echo "Using coupled stats dataset: ${COUPLED_STATS_DATA}"
    else
        echo "Using the following stats:"
        echo " - Atmosphere: ${ATMOS_STATS_DATA}"
        echo " - Ocean: ${OCEAN_STATS_DATA}"
    fi
}

# Get Beaker username from account info
# Outputs: Beaker username to stdout
# Example: BEAKER_USERNAME=$(get_beaker_username)
get_beaker_username() {
    beaker account whoami --format=json | jq -r '.[0].name'
}

# Build job name from group and optional tag
# Args:
#   $1 - JOB_GROUP (required)
#   $2 - TAG (optional)
#   $3 - SUFFIX (optional, defaults to "train")
# Outputs: Job name to stdout
# Example: JOB_NAME=$(build_job_name "$GROUP" "$TAG" "train")
build_job_name() {
    local GROUP="$1"
    local TAG="${2:-}"
    local SUFFIX="${3:-train}"

    if [[ -n "$TAG" ]]; then
        echo "${GROUP}-${TAG}-${SUFFIX}"
    else
        echo "${GROUP}-${SUFFIX}"
    fi
}

# Append experiment result to experiments.txt and commit
# Args:
#   $1 - EXPERIMENT_DIR (e.g., "experiments/2025-08-08-jamesd/coupled")
#   $2 - CONFIG_SUBDIR (e.g., "v2025-06-03-fto")
#   $3 - JOB_GROUP
#   $4 - TAG
#   $5 - EXPERIMENT_ID
#   $6 - STATUS (e.g., "training")
#   $7 - CHECKPOINT (e.g., "best_inference_ckpt")
#   $8 - PRIORITY (e.g., "normal")
#   $9 - PREEMPTIBLE_FLAG (e.g., "--not-preemptible")
#   $10 - GIT_BRANCH
# Example: append_to_experiments_file "$EXPERIMENT_DIR" "$CONFIG_SUBDIR" "$JOB_GROUP" "$TAG" "$EXPERIMENT_ID" "training" "best_inference_ckpt" "normal" "--not-preemptible" "$GIT_BRANCH"
append_to_experiments_file() {
    local EXPERIMENT_DIR="$1"
    local CONFIG_SUBDIR="$2"
    local JOB_GROUP="$3"
    local TAG="$4"
    local EXPERIMENT_ID="$5"
    local STATUS="$6"
    local CHECKPOINT="$7"
    local PRIORITY="$8"
    local PREEMPTIBLE_FLAG="$9"
    local GIT_BRANCH="${10}"

    local REPO_ROOT=$(git rev-parse --show-toplevel)
    local EXPERIMENTS_FILE="$REPO_ROOT/$EXPERIMENT_DIR/$CONFIG_SUBDIR/experiments.txt"

    { echo;
      echo "${JOB_GROUP}|${TAG}|${EXPERIMENT_ID}|${STATUS}|${CHECKPOINT}|${PRIORITY}|${PREEMPTIBLE_FLAG}";
    } >> "$EXPERIMENTS_FILE"

    git_commit_and_push "$EXPERIMENTS_FILE" \
        "Update ${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "$GIT_BRANCH"
}

# Initialize script environment variables
# Sets up REPO_ROOT, GIT_BRANCH, and BEAKER_USERNAME
# These are commonly needed by all wrapper scripts
# Example: init_script_environment
init_script_environment() {
    REPO_ROOT=$(git rev-parse --show-toplevel)
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    BEAKER_USERNAME=$(get_beaker_username)

    # Export so they're available to callers
    export REPO_ROOT
    export GIT_BRANCH
    export BEAKER_USERNAME
}

# Parse dry-run flag from arguments
# Sets global DRY_RUN variable (true/false)
# Usage: parse_dry_run_flag "$@"
parse_dry_run_flag() {
    DRY_RUN=false
    for arg in "$@"; do
        if [[ "$arg" == "--dry-run" ]]; then
            DRY_RUN=true
            break
        fi
    done
    export DRY_RUN
}

# Print dry-run header
print_dry_run_header() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "========================================"
        echo "DRY RUN MODE - No jobs will be launched"
        echo "========================================"
        echo
        echo "Environment:"
        echo "  Repository Root: $REPO_ROOT"
        echo "  Git Branch: $GIT_BRANCH"
        echo "  Beaker Username: $BEAKER_USERNAME"
        print_stats_config
        echo
    fi
}

# Print detailed job info for first job
print_detailed_job_info() {
    echo "----------------------------------------"
    echo "JOB: $JOB_NAME (DETAILED)"
    echo "----------------------------------------"
    echo "Parsed Fields:"
    echo "  GROUP: $GROUP"
    echo "  TAG: ${TAG:-(empty)}"
    echo "  STATUS: $STATUS"
    echo "  PRIORITY: $PRIORITY"
    echo "  CLUSTER: ${CLUSTER:-(default: H100 clusters)}"
    echo "  N_GPUS: $N_GPUS"
    echo "  SHARED_MEM: $SHARED_MEM"
    echo "  RETRIES: $RETRIES"
    echo "  WORKSPACE: $WORKSPACE"
    echo "  OVERRIDE_ARGS: ${OVERRIDE_ARGS:-(none)}"
    echo
    echo "Computed Values:"
    echo "  Job Name: $JOB_NAME"
    echo "  Job Group: $JOB_GROUP"
    echo "  Config Path: ${CONFIG_PATH_REL:-${CONFIG_PATH}}"
    echo "  FME Module: $FME_MODULE"
    echo
}

# Print condensed job summary for subsequent jobs
# Args: JOB_NAME, CONFIG_PATH, CLUSTER, N_GPUS, SHARED_MEM, PRIORITY
print_condensed_job_info() {
    local JOB_NAME="$1"
    local CONFIG_PATH="$2"
    local CLUSTER="$3"
    local N_GPUS="$4"
    local SHARED_MEM="$5"
    local PRIORITY="$6"

    printf "  - %-50s | %s | GPUs: %2s | Mem: %7s | Priority: %-8s | Cluster: %s\n" \
        "$JOB_NAME" "$CONFIG_PATH" "$N_GPUS" "$SHARED_MEM" "$PRIORITY" "${CLUSTER:-(default)}"
}

# Wrapper for gantry job that respects dry-run mode
# Returns mock experiment ID in dry-run mode
run_gantry_training_job_with_dry_run() {
    if [[ "$DRY_RUN" == "true" ]]; then
        # Return mock experiment ID
        echo "01234567890abcdef"
    else
        run_gantry_training_job "$@"
    fi
}

# Wrapper for git operations that respects dry-run mode
git_commit_and_push_with_dry_run() {
    if [[ "$DRY_RUN" != "true" ]]; then
        git_commit_and_push "$@"
    fi
}

# Wrapper for appending to experiments.txt that respects dry-run mode
append_to_experiments_file_with_dry_run() {
    if [[ "$DRY_RUN" != "true" ]]; then
        append_to_experiments_file "$@"
    fi
}

# Print dry-run summary at the end
# Args: TOTAL_JOBS, PROCESSED_JOBS, SKIPPED_JOBS
print_dry_run_summary() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo
        echo "----------------------------------------"
        echo "SUMMARY"
        echo "----------------------------------------"
        echo "Total Jobs in File: $1"
        echo "  - Will Process: $2 (STATUS=train)"
        echo "  - Will Skip: $3 (STATUS!=train)"
        echo
        echo "Actions that WOULD be taken:"
        echo "  - Launch $2 gantry jobs"
        echo "  - Commit $2 config files to git"
        echo "  - Append $2 entries to experiments.txt"
        echo "  - Push git commits to branch: $GIT_BRANCH"
        echo
        echo "========================================"
        echo "DRY RUN COMPLETE - No changes were made"
        echo "========================================"
    fi
}
