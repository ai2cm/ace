#!/opt/homebrew/bin/bash

set -e

CONFIG_FILENAME="equilibrium-climate-data-only-evaluator.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)

CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

MODEL="01KATCK2231JP93DMJ09DY75AM"

for climate in "1xCO2" "2xCO2" "4xCO2"; do
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override loader.dataset.file_pattern=${climate}-ic_0001.zarr prediction_loader.dataset.file_pattern=${climate}-ic_0001.zarr
    job_name="2026-02-06-SHiELD-SOM-${climate}-ic_0001-data-only-inference"
    gantry run \
        --name $job_name \
        --description 'Run ACE data-only equilibrium climate inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/jupiter \
        --cluster ai2/ceres \
        --cluster ai2/saturn \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $MODEL:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH --override loader.dataset.file_pattern=${climate}-ic_0001.zarr prediction_loader.dataset.file_pattern=${climate}-ic_0001.zarr
done
