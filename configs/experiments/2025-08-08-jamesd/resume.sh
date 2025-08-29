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

# Since we use a service account API key for wandb, we use the beaker username to set the wandb username.
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# FIXME: this needs to be per-task configurable
ATMOS_STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-atmosphere
OCEAN_STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean

# Change to the repo root so paths are valid no matter where we run the script from.
cd "$REPO_ROOT"

while read RESUMING; do
    GROUP=$(echo "$RESUMING" | cut -d"|" -f1)
    WANDB_PROJECT=$(echo "$RESUMING" | cut -d"|" -f2)
    WANDB_ID=$(echo "$RESUMING" | cut -d"|" -f3)
    STATUS=$(echo "$RESUMING" | cut -d"|" -f4)
    PRIORITY=$(echo "$RESUMING" | cut -d"|" -f5)
    CLUSTER=$(echo "$RESUMING" | cut -d"|" -f6)
    N_GPUS=$(echo "$RESUMING" | cut -d"|" -f7)
    SHARED_MEM=$(echo "$RESUMING" | cut -d"|" -f8)
    RETRIES=$(echo "$RESUMING" | cut -d"|" -f9)
    WORKSPACE=$(echo "$RESUMING" | cut -d"|" -f10)
    OVERRIDE_ARGS=$(echo "$RESUMING" | cut -d"|" -f11)
    # can be used in place of WANDB_PROJECT and WANDB_ID
    EXISTING_RESULTS_DATASET=$(echo "$RESUMING" | cut -d"|" -f12)
    if [[ "$STATUS" != "train" ]]; then
        continue
    fi
    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi
    JOB_GROUP="${GROUP}"
    JOB_NAME="${JOB_GROUP}-train"
    # Determine which fme module to use based on CONFIG_SUBDIR
    if [[ "$CONFIG_SUBDIR" =~ ^coupled ]]; then
        FME_MODULE="fme.coupled.train"
    else
        FME_MODULE="fme.ace.train"
    fi
    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXPER_ID=$(
            python $REPO_ROOT/scripts/wandb/wandb_to_beaker_experiment.py \
              --project "$WANDB_PROJECT" --wandb_id "$WANDB_ID"
        )
        EXISTING_RESULTS_DATASET=$(
            beaker experiment get $EXPER_ID --format json |
              jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4
        )
    fi
    declare -a CLUSTER_ARGS
    if [[ "$CLUSTER" == "titan" ]]; then
        if [[ -z "$WORKSPACE" ]]; then
            WORKSPACE=ai2/climate-titan
        fi
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster ai2/titan-cirrascale
        )
    else
        if [[ -z "$WORKSPACE" ]]; then
            WORKSPACE=ai2/climate-ceres
        fi
        CLUSTER_ARGS=(
            --workspace "$WORKSPACE"
            --cluster ai2/ceres-cirrascale
        )
    fi

    echo
    echo "Resuming ${CONFIG_SUBDIR} training job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Resuming results dataset ID: ${EXISTING_RESULTS_DATASET}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - Workspace: ${WORKSPACE}"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Override: ${OVERRIDE_ARGS}"

    EXPERIMENT_ID=$(
        gantry run \
            --name "$JOB_NAME" \
            --description "Resume ${CONFIG_SUBDIR} pretraining: ${JOB_GROUP}" \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --priority "$PRIORITY" \
            --preemptible \
            --retries $RETRIES \
            "${CLUSTER_ARGS[@]}" \
            --weka climate-default:/climate-default \
            --env WANDB_USERNAME=$BEAKER_USERNAME \
            --env WANDB_NAME=$JOB_NAME \
            --env WANDB_JOB_TYPE=training \
            --env WANDB_RUN_GROUP=$JOB_GROUP \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset $ATMOS_STATS_DATA:/atmos_stats \
            --dataset $OCEAN_STATS_DATA:/ocean_stats \
            --dataset "$EXISTING_RESULTS_DATASET:/existing-results" \
            --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/ckpt.tar:/ckpt.tar \
            --gpus "${N_GPUS}" \
            --shared-memory "${SHARED_MEM}" \
            --budget ai2/climate \
            --no-conda \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node "$N_GPUS" -m $FME_MODULE /existing-results/config.yaml \
            --override resume_results.existing_dir=/existing-results $OVERRIDE_ARGS |
            tee /dev/tty |
            grep beaker.org |
            cut -d/ -f5
    )

    # remove or change 'training' once completed in order to submit an evaluator job
    { echo;
      echo "${JOB_GROUP}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git add "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"
    git commit -m"Update ${CONFIG_SUBDIR}/experiments.txt"
    git push origin "${GIT_BRANCH}"

done <"${SCRIPT_DIR}/${CONFIG_SUBDIR}/resuming.txt"
