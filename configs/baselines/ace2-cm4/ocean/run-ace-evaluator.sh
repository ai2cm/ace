#!/bin/bash

set -e

CONFIG_FILENAME="ace-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

while read TRAIN_EXPER; do
    JOB_GROUP=$(echo "$TRAIN_EXPER" | cut -d'|' -f1)
    RS=$(echo "$TRAIN_EXPER" | cut -d'|' -f2)
    EXPER_ID=$(echo "$TRAIN_EXPER" | cut -d'|' -f3)
    STATUS=$(echo "$TRAIN_EXPER" | cut -d'|' -f4)
    CKPT=$(echo "$TRAIN_EXPER" | cut -d"|" -f5)
    OVERRIDE_ARGS=$(echo "$TRAIN_EXPER" | cut -d"|" -f6)
    if [[ "$STATUS" == "training" ]] || [[ "$STATUS" == "skip" ]]; then
        continue
    fi

    EXISTING_RESULTS_DATASET=$(beaker experiment get $EXPER_ID --format json | jq '.[].jobs[-1].result' | grep "beaker" | cut -d'"' -f4)
    echo
    echo "Launching evaluator job:"
    echo " - Group: ${JOB_GROUP}"
    echo " - Random seed iteration: ${RS}"
    echo " - Checkpoint: ${CKPT}"
    echo " - Training experiment ID: ${EXPER_ID}"
    echo " - Training results dataset ID: ${EXISTING_RESULTS_DATASET}"
    echo " - --override args: ${OVERRIDE_ARGS}"

    echo
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override $OVERRIDE_ARGS
    echo

    JOB_NAME="${JOB_GROUP}-evaluator_${CKPT}-rs${RS}"
    DESCRIPTION="ACE-Samudra CM4 baseline evaluator of RS${RS} ${CKPT}"
    gantry run \
        --name "${JOB_NAME}" \
        --description "${DESCRIPTION}" \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/ceres-cirrascale \
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
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --no-conda \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH --override $OVERRIDE_ARGS
    echo
done <"${SCRIPT_PATH}/experiments.txt"
