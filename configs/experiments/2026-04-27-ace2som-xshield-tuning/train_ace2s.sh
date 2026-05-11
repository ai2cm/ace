#!/bin/bash

set -e


CONFIG_FILENAME="tune-ace2s-v1-xshield-1yr-4k.yaml"

SCRIPT_PATH=$(git rev-parse --show-prefix)
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_GROUP=ace
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
STATS_DATASET=andrep/2025-09-16-X-SHiELD-AMIP-1deg-8layer-11yr-stats
PRE_TRAINED_WEIGHTS_PATH=/pre-trained-weights/training_checkpoints/best_ckpt.tar
SEED_OFFSET=10

cd $REPO_ROOT


# ACE2S ckpt used in paper
ACE2S_CKPT=("01KQDG7X72D4E2JJTGQ0ZF9J9T")


# --dataset ${PRE_TRAINED_WEIGHTS_DATASETS[$seed]}:training_checkpoints/best_ckpt.tar:/ckpt.tar \

#       --dataset $TUNED_DATASET:/pre-trained-weights \
#        --dataset ${PRE_TRAINED_WEIGHTS_DATASETS[$seed]}:/pre-trained-weights \
for seed in {0..0}; do
    #job_name="ace2som-xshield-tune-1yr-even-split-single-decoder-seed${seed}"
    job_name="ace2som-xshield-tune-1yr-4k-seed${seed}"
    fine_tune_seed=$((seed + SEED_OFFSET))
    override="seed=${fine_tune_seed}"
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
        --env-secret WANDB_API_KEY=wandb-api-key-annak \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $STATS_DATASET:/statsdata \
        --dataset ${ACE2S_CKPT[$seed]}:training_checkpoints/ACE2S.ckpt:/ckpt.tar \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --no-python \
        --install "pip install --no-deps ." \
        --allow-dirty \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
done
