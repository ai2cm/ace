#!/bin/bash

set -e

# Configuration variables
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
# Since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

# Job configuration
JOB_GROUP="resume_results"  # Update this to match your experiment group
JOB_NAME="${JOB_GROUP}-train"
EXISTING_RESULTS_DATASET="01K3SEWJ5S6VFPN54DH4PF36RB"  # Update this with your results dataset ID
STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean
N_GPUS=8
SHARED_MEM="600GiB"
PRIORITY="urgent" # FIXME
WORKSPACE="ai2/climate-ceres" # FIXME
RETRIES=0

# Override arguments (add any config overrides here)
OVERRIDE_ARGS="max_epochs=200"

# Change to the repo root so paths are valid no matter where we run the script from
cd "$REPO_ROOT"

echo
echo "Resuming training job:"
echo " - Job name: ${JOB_NAME}"
echo " - Resuming results dataset ID: ${EXISTING_RESULTS_DATASET}"
echo " - Priority: ${PRIORITY}"
echo " - GPUs: ${N_GPUS}"
echo " - Shared memory: ${SHARED_MEM}"
echo " - Override args: ${OVERRIDE_ARGS}"

gantry run \
    --name "$JOB_NAME" \
    --description "Resume training: ${JOB_GROUP}" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace $WORKSPACE \
    --priority "$PRIORITY" \
    --preemptible \
    --retries $RETRIES \
    --cluster ai2/ceres-cirrascale \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $STATS_DATA:/statsdata \
    --dataset "$EXISTING_RESULTS_DATASET:/existing-results" \
    --gpus "$N_GPUS" \
    --shared-memory "$SHARED_MEM" \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train /existing-results/config.yaml \
    --override resume_results.existing_dir=/existing-results $OVERRIDE_ARGS
