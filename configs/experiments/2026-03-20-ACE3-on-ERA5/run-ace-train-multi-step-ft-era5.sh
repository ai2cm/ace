#!/bin/bash

set -e

CONFIG_FILENAME="multi-step-fine-tune-config-era5-energy-corrector.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_GROUP=ace
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
STATS_DATASET=andrep/2026-02-06-vertically-resolved-1deg-c96-shield-ramped-climSST-random-CO2-ensemble-fme-dataset-stats
PRE_TRAINED_WEIGHTS_PATH=/pre-trained-weights/training_checkpoints/best_ckpt.tar
SEED_OFFSET=10

cd $REPO_ROOT

declare -A PRE_TRAINED_WEIGHTS_DATASETS=( \
    [0]="01KHJ5F1M6YKVZESPZAAVVD6G8" \
    [1]="01KHCEF1SBYCZCGDM78N1CJC3H" \
)

# Simple list of learning rates
LEARNING_RATES=(1e-4)  #1e-6)

for seed in 0; do
    for lr in "${LEARNING_RATES[@]}"; do

        job_name="ace-era5-pt-multi-step-shield-ft-energy-correctors-on-lr${lr}-rs${seed}-sampler"

        fine_tune_seed=$((seed + SEED_OFFSET))

        override="seed=${fine_tune_seed} \
stepper_training.parameter_init.weights_path=${PRE_TRAINED_WEIGHTS_PATH} \
optimization.lr=${lr}"

        python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override

        gantry run \
            --name $job_name \
            --description 'Run ACE training' \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --workspace ai2/climate-titan \
            --priority urgent \
            --preemptible \
            --cluster ai2/titan \
            --env WANDB_NAME=$job_name \
            --env WANDB_USERNAME=$WANDB_USERNAME \
            --env WANDB_JOB_TYPE=training \
            --env WANDB_RUN_GROUP=$WANDB_GROUP \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset $STATS_DATASET:/statsdata \
            --dataset ${PRE_TRAINED_WEIGHTS_DATASETS[${seed}]}:/pre-trained-weights \
            --gpus $N_GPUS \
            --shared-memory 400GiB \
            --weka climate-default:/climate-default \
            --budget ai2/climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override

    done
done
