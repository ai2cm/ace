#!/bin/bash

set -e


CONFIG_FILENAME="tune-era5-on-xshield-10yr-old-weights.yaml"

SCRIPT_PATH=$(git rev-parse --show-prefix)
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-${BEAKER_USERNAME}}
WANDB_GROUP=ace
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
STATS_DATASET=andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019
#STATS_DATASET=andrep/2025-09-11-X-SHiELD-AMIP-1deg-8layer-11yr-stats
SEED_OFFSET=10

cd $REPO_ROOT

# ERA5 pretraining https://beaker.org/orgs/ai2/workspaces/ace/work/01KSN76D3GQ7MVP058Y2Z2TGKE
PRE_TRAINED_WEIGHTS_DATASETS=("01KSVC6YS7C18SGYV4VPZYZ232")

# ERA5 pretraining, different random seed, with energy correction and embed_dim 32
#PRE_TRAINED_WEIGHTS_DATASETS=("01KVZV0DFM43B7XREKTYK210VX")

for seed in {0..0}; do
    #job_name="ace2som-xshield-tune-1yr-even-split-single-decoder-seed${seed}"
    job_name="ace2s-era5-tune-xshield-10yr-old-weights-seed${seed}"
    fine_tune_seed=$((seed + SEED_OFFSET))
    override="seed=${fine_tune_seed}"
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override

    gantry run \
        --name $job_name \
        --description 'Run ACE training' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/titan \
        --env WANDB_NAME=$job_name \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$WANDB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-annak \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset ${PRE_TRAINED_WEIGHTS_DATASETS[$seed]}:training_checkpoints/best_ckpt.tar:/ckpt.tar \
        --dataset $STATS_DATASET:/statsdata \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --no-python \
        --install "pip install --no-deps ." \
        --allow-dirty \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $override
done
