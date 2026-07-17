#!/bin/bash

set -e

# Fine-tune a trainable pressure-level secondary decoder on top of the FROZEN
# nobgd4ek core (1deg v2, no-residual, NO-CO2). See the header of
# ace-fine-tune-pressure-level-separate-decoder-nobgd4ek-config.yaml for the three
# launch blockers (code-generation rebase, merged normalization stats, daily-06Z
# plev target). DO NOT launch until those are resolved.

JOB_NAME_BASE="ace-aimip-nobgd4ek-fine-tune-decoder-pressure-levels"
JOB_GROUP="ace-aimip-nobgd4ek"
CONFIG_FILENAME="ace-fine-tune-pressure-level-separate-decoder-nobgd4ek-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
# nobgd4ek: 1deg v2 no-residual no-CO2 checkpoint (best_inference_ckpt.tar)
EXISTING_RESULTS_DATASET="01KXKBKW2DCAYX6Q3FRJ60K3Q6"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    CONFIG_FILENAME=$2
    shift 2
    OVERRIDE="$@"

    gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Fine-tune nobgd4ek ACE decoder outputs on AIMIP period' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2cm/ace \
        --priority high \
        --preemptible \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/titan-cirrascale \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=training \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset brianhenn/era5-1deg-8layer-pressure-level-stats-1990-2019-v2:/statsdata \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/base_weights/ckpt.tar \
        --gpus $N_GPUS \
        --shared-memory 400GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_FILENAME --override $OVERRIDE

}

# NOTE: the /statsdata mount above is brianhenn plev-stats-v2, whose CORE-variable
# stats do NOT match nobgd4ek (see blocker 2). Replace it with a merged stats dataset
# (nobgd4ek 2026-03-19 daily core stats + plev stats) before launching.

# Single seed (RS0). The frozen core is identical across seeds; only the decoder init
# and data shuffling vary, so one seed is sufficient for a first look.
for SEED in 0; do
    JOB_NAME="${JOB_NAME_BASE}-separate-decoder-lr-warmup-RS${SEED}"
    OVERRIDE="seed=${SEED}"
    launch_job $JOB_NAME $CONFIG_PATH $OVERRIDE
done
