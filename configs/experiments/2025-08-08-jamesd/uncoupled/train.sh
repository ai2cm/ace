#!/bin/bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <config_subdirectory>"
  echo "  - <config_subdirectory>: Subdirectory containing the 'train-config.yaml' to use."
  exit 1
fi

# The subdirectory (passed as an argument) that holds the config file.
CONFIG_SUBDIR=$1

# Get the absolute directory where this script is located.
SCRIPT_DIR=$(git rev-parse --show-prefix)

# Construct the full path to the specified configuration file.
CONFIG_FILENAME="train-config.yaml"
CONFIG_PATH="$SCRIPT_DIR/$CONFIG_SUBDIR/$CONFIG_FILENAME"

# Since we use a service account API key for wandb, we use the beaker username to set the wandb username.
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# FIXME: this needs to be per-task configurable
ATMOS_STATS_DATA=jamesd/2025-08-14-cm4-piControl-200yr-coupled-stats-atmosphere
OCEAN_STATS_DATA=jamesd/2025-08-14-cm4-piControl-200yr-coupled-stats-ocean

# Change to the repo root so paths are valid no matter where we run the script from.
cd "$REPO_ROOT"

while read TRAINING; do
    GROUP=$(echo "$TRAINING" | cut -d"|" -f1)
    STATUS=$(echo "$TRAINING" | cut -d"|" -f2)
    PRIORITY=$(echo "$TRAINING" | cut -d"|" -f3)
    CLUSTER=$(echo "$TRAINING" | cut -d"|" -f4)
    N_GPUS=$(echo "$TRAINING" | cut -d"|" -f5)
    SHARED_MEM=$(echo "$TRAINING" | cut -d"|" -f6)
    RETRIES=$(echo "$TRAINING" | cut -d"|" -f7)
    OVERRIDE_ARGS=$(echo "$TRAINING" | cut -d"|" -f8)
    if [[ "$STATUS" != "train" ]]; then
        continue
    fi
    if [[ -z $RETRIES ]]; then
        RETRIES=0
    fi
    JOB_GROUP="${GROUP}"
    JOB_NAME="${JOB_GROUP}-train"
    declare -a CLUSTER_ARGS
    if [[ "$CLUSTER" == "titan" ]]; then
        CLUSTER_ARGS=(
            --workspace ai2/climate-titan
            --cluster ai2/titan-cirrascale
        )
    else
        CLUSTER_ARGS=(
            --workspace ai2/climate-ceres
            --cluster ai2/ceres-cirrascale
        )
    fi

    # get the template from the config subdir
    if [[ -n "${OVERRIDE_ARGS}" ]]; then
        OVERRIDE="--override ${OVERRIDE_ARGS}"
    else
        OVERRIDE=""
    fi

    echo
    echo "Launching uncoupled training job:"
    echo " - Job name: ${JOB_NAME}"
    echo " - Config: ${CONFIG_PATH}"
    echo " - Priority: ${PRIORITY}"
    echo " - Cluster: ${CLUSTER} (${RETRIES} retries)"
    echo " - GPUs: ${N_GPUS}"
    echo " - Shared memory: ${SHARED_MEM}"
    echo " - Override: ${OVERRIDE_ARGS}"

    python -m fme.ace.validate_config "$CONFIG_PATH" --config_type train $OVERRIDE

    if git status --porcelain "$CONFIG_PATH" | grep -q .; then
        git add "$CONFIG_PATH"
        git commit -m"${JOB_NAME}"
        git push origin "${GIT_BRANCH}"
    fi
    echo

    EXPERIMENT_ID=$(
        gantry run \
            --name $JOB_NAME \
            --description "Run uncoupled pretraining: ${JOB_GROUP}" \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --priority $PRIORITY \
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
            --gpus "${N_GPUS}" \
            --shared-memory "${SHARED_MEM}" \
            --budget ai2/climate \
            --no-conda \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train "$CONFIG_PATH" $OVERRIDE |
            tee /dev/tty |
            grep beaker.org |
            cut -d/ -f5
    )

    # remove or change 'training' once completed in order to submit an evaluator job
    { echo;
      echo "${JOB_GROUP}|${EXPERIMENT_ID}|training|best_inference_ckpt|normal|--not-preemptible";
    } >> "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"

    git add "${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"
    git commit -m"Update uncoupled/${CONFIG_SUBDIR}/experiments.txt"
    git push origin "${GIT_BRANCH}"

done <"${SCRIPT_DIR}/${CONFIG_SUBDIR}/training.txt"
