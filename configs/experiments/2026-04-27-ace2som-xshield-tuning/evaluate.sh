#!/bin/bash

# uses best_ckpt.tar instead of best_inference_ckpt.tar to avoid q0
# determining best ckpt. this is effectively the last ckpt from tuning

set -e


CONFIG_FILENAME="ace-evaluator-4k.yaml"

SCRIPT_PATH=$(git rev-parse --show-prefix)
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_GROUP=ace
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT

MODEL_CHECKPOINT_DATASETS=("01KQD8NF9HQD1QY2X0S132YH72" "01KQD8NMEYCVQ835WQV751MNYP")


for seed in {1..1}; do
    job_name="evaluate-4k-ace2som-xshield-tune-1yr-even-split-single-decoder-seed${seed}"
    gantry run \
        --name $job_name \
        --description 'Run ACE training' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --preemptible \
        --cluster ai2/titan \
        --cluster ai2/jupiter \
        --env WANDB_NAME=$job_name \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$WANDB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-annak \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset ${MODEL_CHECKPOINT_DATASETS[$seed]}:training_checkpoints/best_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --no-python \
        --install "pip install --no-deps ." \
        --allow-dirty \
        --no-python \
        -- python -m fme.ace.evaluator $CONFIG_PATH
done