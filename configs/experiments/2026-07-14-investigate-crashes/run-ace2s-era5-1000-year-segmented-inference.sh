#!/opt/homebrew/bin/bash

set -e

DATE="2026-08-14"
WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-era5-1000-year-inference-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

DATASET_ID=01KWD8DZVJFKYC5A9PNW8259GH

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

for seed in {0..0}; do
    override="\
        n_forward_steps=182621 \
        seed=$seed \
    "
    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $override

    job_name="${DATE}-ace2s-era5-1000-year-inference-seed-${seed}"
    gantry run \
        --name $job_name \
        --description 'Run inference with ACE' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority urgent \
        --cluster ai2/jupiter \
        --env CM_PRIORITY=high \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $DATASET_ID:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --system-python \
        --min-runtime 8h \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH --override $override --segments 8
done
