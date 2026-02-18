#!/opt/homebrew/bin/bash

set -e

DATE="2026-02-18"
CONFIG_FILENAME="ace-amip-data-only-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)

CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

REFERENCE_MODEL="01KHGDAMB2BDZQS8JFF65A2YDR"

for ensemble_id in "ic_0001" "ic_0002"; do
    override="loader.dataset.file_pattern=${ensemble_id}.zarr prediction_loader.dataset.file_pattern=${ensemble_id}.zarr"
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override $override
    job_name="${DATE}-amip-${ensemble_id}-data-only-evaluator"
    gantry run \
        --name $job_name \
        --description 'Run ACE AMIP data-only evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/jupiter \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $REFERENCE_MODEL:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH --override $override
done
