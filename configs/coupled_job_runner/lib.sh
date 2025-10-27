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
}

# Set default values for stats datasets if not provided via CLI args
# Uses defaults: jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-{atmosphere,ocean}
set_default_stats() {
    if [[ -z "$COUPLED_STATS_DATA" ]]; then
        if [[ -z "$ATMOS_STATS_DATA" ]]; then
            ATMOS_STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-atmosphere
        fi
        if [[ -z "$OCEAN_STATS_DATA" ]]; then
            OCEAN_STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean
        fi
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
            --dataset "$COUPLED_STATS_DATA:uncoupled_ocean:/ocean_stats"
        )
    else
        STATS_DATASET_ARGS=(
            --dataset "$ATMOS_STATS_DATA:/atmos_stats"
            --dataset "$OCEAN_STATS_DATA:/ocean_stats"
        )
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
        OVERRIDE="$OVERRIDE_ARGS"
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
            grep beaker.org/ex |
            cut -d/ -f5
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
