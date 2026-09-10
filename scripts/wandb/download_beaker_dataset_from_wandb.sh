#!/bin/bash

# Download a Beaker dataset by looking up the experiment from wandb.
#
# Usage:
#   download_beaker_dataset_from_wandb.sh [OPTIONS]
#
# Required Options:
#   --project    wandb project name
#   --jobid        wandb run ID
#   --output-dir   Local directory to download the dataset to
#
# Optional:
#   --entity       wandb entity (default: ai2cm)
#   --prefix       Only download files starting with this prefix
#
# Example:
#   bash download_beaker_dataset_from_wandb.sh \
#       --project my-project \
#       --jobid abc123xyz \
#       --output-dir /path/to/output \
#       --prefix checkpoints/
#
# The dataset will be downloaded to: <output-dir>/<entity>/<project>/runs/<jobid>/

# Get beaker dataset ID from experiment ID
# Args: $1 = EXPERIMENT_ID
# Outputs: dataset ID to stdout
get_beaker_dataset_from_experiment() {
    local EXPER_ID="$1"
    beaker experiment get "$EXPER_ID" --format json |
        jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
}

get_experiment_from_wandb() {
    local ENTITY="$1"
    local PROJECT="$2"
    local WANDB_ID="$3"
    local REPO_ROOT=$(git rev-parse --show-toplevel)

    python "$REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py" \
        --entity "$ENTITY" --project "$PROJECT" --wandb_id "$WANDB_ID"
}

# Default values
ENTITY="ai2cm"

# Parse command line arguments
while [[ "$#" -gt 0 ]]
do case $1 in
    --entity) ENTITY="$2"
    shift;;
    --project) PROJECT="$2"
    shift;;
    --jobid) JOBID="$2"
    shift;;
    --output-dir) OUTPUT_DIR="$2"
    shift;;
    --prefix) PREFIX="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

# Validate required arguments
if [[ -z "${PROJECT}" ]]; then
    echo "Option --project is required"
    exit 1
fi

if [[ -z "${JOBID}" ]]; then
    echo "Option --jobid is required"
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "Option --output-dir is required"
    exit 1
fi

# Get experiment ID from wandb
EXPERIMENT_ID=$(get_experiment_from_wandb "$ENTITY" "$PROJECT" "$JOBID")

# Get dataset ID from experiment
DATASET_ID=$(get_beaker_dataset_from_experiment "$EXPERIMENT_ID")

# Fetch dataset to output directory
if [[ -n "${PREFIX}" ]]; then
    beaker dataset fetch "$DATASET_ID" --output "$OUTPUT_DIR/$ENTITY/$PROJECT/runs/$JOBID" --prefix "$PREFIX"
else
    beaker dataset fetch "$DATASET_ID" --output "$OUTPUT_DIR/$ENTITY/$PROJECT/runs/$JOBID"
fi
