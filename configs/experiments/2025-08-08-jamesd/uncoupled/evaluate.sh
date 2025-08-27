#!/bin/bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <config_subdirectory> [config_filename]"
  exit 1
fi

# The subdirectory (passed as an argument) that holds the config file.
CONFIG_SUBDIR=$1

# Get the absolute directory where this script is located.
SCRIPT_DIR=$(git rev-parse --show-prefix)

 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

while read TRAIN_EXPER; do
    JOB_GROUP=$(echo "$TRAIN_EXPER" | cut -d"|" -f1)
    EXPER_ID=$(echo "$TRAIN_EXPER" | cut -d"|" -f2)
    STATUS=$(echo "$TRAIN_EXPER" | cut -d"|" -f3)
    CKPT=$(echo "$TRAIN_EXPER" | cut -d"|" -f4)
    PRIORITY=$(echo "$TRAIN_EXPER" | cut -d"|" -f5)
    PREEMPTIBLE=$(echo "$TRAIN_EXPER" | cut -d"|" -f6)
    OVERRIDE_ARGS=$(echo "$TRAIN_EXPER" | cut -d"|" -f7)
    # can be used in place of EXPER_ID in case the final results dataset is not desired
    EXISTING_RESULTS_DATASET=$(echo "$TRAIN_EXPER" | cut -d"|" -f8)
    # Check if STATUS starts with "run_"
    if [[ ! "$STATUS" =~ ^run_ ]]; then
        continue
    fi
    # Derive config tag and filename from STATUS
    # Example: if STATUS is "run_ICx1"
    # CURRENT_CONFIG_TAG becomes "ICx1"
    # CURRENT_CONFIG_FILENAME becomes "evaluator-config-ICx1.yaml"
    CURRENT_CONFIG_TAG=${STATUS#run_}
    CURRENT_CONFIG_FILENAME="evaluator-config-${CURRENT_CONFIG_TAG}.yaml"

    JOB_NAME="${JOB_GROUP}-evaluator_${CKPT}-${CURRENT_CONFIG_TAG}"

    # Construct the full path to the configuration file for the current experiment
    CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_SUBDIR}/${CURRENT_CONFIG_FILENAME}"

    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "Error: Config file not found at ${REPO_ROOT}/${CONFIG_PATH} for JOB_NAME: ${JOB_NAME}. Skipping."
        continue
    fi
    if [[ -z $PRIORITY ]]; then
        PRIORITY=normal
    fi
    if [[ -z $PREEMPTIBLE ]]; then
        PREEMPTIBLE=--not-preemptible
    fi
    if [[ -z $EXISTING_RESULTS_DATASET ]]; then
        EXISTING_RESULTS_DATASET=$(beaker experiment get $EXPER_ID --format json | jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4)
    fi

    echo
    echo "Launching uncoupled evaluator job:"
    echo " - Config path: ${CONFIG_PATH}"
    echo " - Group: ${JOB_GROUP}"
    echo " - Checkpoint: ${CKPT}"
    echo " - Training results dataset ID: ${EXISTING_RESULTS_DATASET}"
    echo " - Priority: ${PRIORITY}"
    echo " - ${PREEMPTIBLE}"
    echo " - --override args: ${OVERRIDE_ARGS}"

    echo

    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override $OVERRIDE_ARGS

    echo $JOB_NAME
    gantry run \
        --name $JOB_NAME \
        --description 'Run uncoupled evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-ceres \
        --priority $PRIORITY \
        $PREEMPTIBLE \
        --cluster ai2/ceres-cirrascale \
        --weka climate-default:/climate-default \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "${EXISTING_RESULTS_DATASET}:training_checkpoints/${CKPT}.tar:/ckpt.tar" \
        --gpus 1 \
        --shared-memory 20GiB \
        --budget ai2/climate \
        --no-conda \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH --override $OVERRIDE_ARGS
    echo
done <"${SCRIPT_DIR}/${CONFIG_SUBDIR}/experiments.txt"
