#!/bin/bash
# Wrapper script for resuming training jobs
# Usage: resume.sh <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared library
source "$SCRIPT_DIR/lib.sh"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <experiment_dir> <config_subdirectory> [--atmos_stats <path>] [--ocean_stats <path>] [--coupled_stats <path>]"
  echo "  - <experiment_dir>: Path to experiment directory (e.g., experiments/2025-08-08-jamesd/coupled or experiments/2025-08-08-jamesd/uncoupled)"
  echo "  - <config_subdirectory>: Subdirectory containing the 'resuming.txt' file."
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

# Change to repo root so paths are valid
cd "$REPO_ROOT"

while read RESUMING; do
    GROUP=$(echo "$RESUMING" | cut -d"|" -f1)
    TAG=$(echo "$TRAINING" | cut -d"|" -f2)
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

    # Run the job - note special handling for resume
    EXPERIMENT_ID=$(
        gantry run \
            --name "$JOB_NAME" \
            --description "Resume ${EXPERIMENT_DIR} pretraining: ${JOB_GROUP}" \
            --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
            --priority "$PRIORITY" \
            --preemptible \
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
            --dataset "$EXISTING_RESULTS_DATASET:/existing-results" \
            --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/ckpt.tar:/ckpt.tar" \
            --gpus "$N_GPUS" \
            --shared-memory "$SHARED_MEM" \
            --budget ai2/climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node "$N_GPUS" -m $FME_MODULE /existing-results/config.yaml \
            --override resume_results.existing_dir=/existing-results $OVERRIDE_ARGS |
            tee /dev/tty |
            grep beaker.org/ex |
            cut -d/ -f5
    )

    # Append to experiments.txt
    { echo;
      echo "${JOB_GROUP}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git_commit_and_push "${FULL_EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "Update ${EXPERIMENT_DIR}/${CONFIG_SUBDIR}/experiments.txt" \
        "$GIT_BRANCH"

done <"$INPUT_PATH"
