#!/opt/homebrew/bin/bash

set -e

CONFIG_FILENAME="ace-abrupt-4xCO2-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)

CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

declare -A MODELS=( \
    [stochastic-ff-ec-eii-0]="01KATCK2231JP93DMJ09DY75AM" \
    [stochastic-ff-ec-eii-1]="01KASWYADJRPS02V61XS970768" \
    [stochastic-rp-ec-eii-0]="01KAVW4WJ4F8M939G1WEG31WSX" \
    [stochastic-rp-ec-eii-1]="01KATW2MKCEAK9EAMWNM7XWH4B" \
)

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH
for name in "${!MODELS[@]}"; do
    job_name="2026-01-26-stochastic-ace-abrupt-4xCO2-evaluator-$name"
    existing_results_dataset=${MODELS[$name]}
    gantry run \
        --name $job_name \
        --description 'Run ACE abrupt 4xCO2 evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --not-preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $existing_results_dataset:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH
done
