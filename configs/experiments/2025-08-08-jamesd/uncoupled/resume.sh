#!/bin/bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <config_subdirectory>"
  echo "  - <config_subdirectory>: Subdirectory containing the 'resuming.txt' file."
  exit 1
fi

# The subdirectory (passed as an argument) that holds the config file.
CONFIG_SUBDIR=$1

# Get the absolute directory where this script is located.
SCRIPT_DIR=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)

# Since we use a service account API key for wandb, we use the beaker username to set the wandb username.
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

# FIXME: this needs to be per-task configurable
ATMOS_STATS_DATA=jamesd/2025-08-14-cm4-piControl-200yr-coupled-stats-atmosphere
OCEAN_STATS_DATA=jamesd/2025-08-14-cm4-piControl-200yr-coupled-stats-ocean

# Change to the repo root so paths are valid no matter where we run the script from.
cd "$REPO_ROOT"

RESUMING_FILE="${SCRIPT_DIR}/${CONFIG_SUBDIR}/resuming.txt"

while read RESUMING; do
    GROUP=$(echo "$FINETUNING" | cut -d"|" -f1)
    WANDB_PROJECT=$(echo "$FINETUNING" | cut -d"|" -f2)
    WANDB_ID=$(echo "$FINETUNING" | cut -d"|" -f3)
    STATUS=$(echo "$FINETUNING" | cut -d"|" -f4)
    PRIORITY=$(echo "$FINETUNING" | cut -d"|" -f5)
    CLUSTER=$(echo "$FINETUNING" | cut -d"|" -f6)
    N_GPUS=$(echo "$FINETUNING" | cut -d"|" -f7)
    SHARED_MEM=$(echo "$FINETUNING" | cut -d"|" -f8)
    RETRIES=$(echo "$FINETUNING" | cut -d"|" -f9)
    OVERRIDE_ARGS=$(echo "$FINETUNING" | cut -d"|" -f10)
    if [[ "$STATUS" != "resume" ]]; then
        echo "Skipping experiment ID ${EXPER_ID} with status '${STATUS}'."
        echo
        continue
    fi
    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi

    EXPER_ID=$(
        python $REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py \
          --project "$WANDB_PROJECT" --wandb_id "$WANDB_ID"
    )
    # Get the results dataset from the last job of the experiment
    EXISTING_RESULTS_DATASET=$(
        beaker experiment get $EXPER_ID --format json |
          jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
    )
    if [[ -z "$EXISTING_RESULTS_DATASET" || "$EXISTING_RESULTS_DATASET" == "null" ]]; then
          echo "ERROR: Could not find results dataset for experiment ${EXPER_ID}. Skipping."
          echo
          continue
    fi
    declare -a CLUSTER_ARGS
    if [[ "$CLUSTER" == "jupiter" ]]; then
        CLUSTER_ARGS=(
            --workspace ai2/ace
            --cluster ai2/jupiter-cirrascale-2
        )
    elif [[ "$CLUSTER" == "ceres" ]]; then
        CLUSTER_ARGS=(
            --workspace ai2/ace # Or a ceres-specific workspace if you have one
            --cluster ai2/ceres-cirrascale
        )
    else
        echo "ERROR: Unknown cluster '${CLUSTER}' for experiment ${EXPER_ID}. Skipping."
        echo
        continue
    fi

    if [[ -n "${OVERRIDE_ARGS}" ]]; then
        OVERRIDE="--override ${OVERRIDE_ARGS}"
    else
        OVERRIDE=""
    fi

    echo
    echo "Resuming uncoupled training job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Original Experiment ID: ${EXPER_ID}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER}"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - New max epochs: ${NEW_MAX_EPOCHS}"
    echo " - Additional overrides: ${OVERRIDE_ARGS:-None}"
    echo

    gantry run \
        --name "$JOB_NAME" \
        --description "Resume uncoupled pretraining: ${JOB_GROUP}" \
        --beaker-image annak/fme-deps-only-d51944c4 \
        --priority "$PRIORITY" \
        --preemptible \
        "${CLUSTER_ARGS[@]}" \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_JOB_TYPE=training \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset jamesd/2025-03-18-cm4-piControl-ocean-atmos-5daily-stats:/statsdata \
        --dataset "$EXISTING_RESULTS_DATASET:/existing-results" \
        --gpus "$N_GPUS" \
        --shared-memory "$SHARED_MEM" \
        --no-conda \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train /existing-results/config.yaml \
        --override "${ALL_OVERRIDES[@]}"

    echo "Job submitted for ${EXPER_ID}."
    echo "-----------------------------------------------------"
    echo
    sleep 1
done <"$RESUMING_FILE"

echo "All specified experiments have been processed."
