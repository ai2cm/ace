#!/bin/bash
#
# SamudrACE CM4 piControl training stage 2: starting from the stage 1
# checkpoint, fine-tunes both atmosphere and ocean models jointly with a
# cosine-annealing LR schedule.

set -e

JOB_NAME="cm4-piControl-coupled-finetune"
JOB_GROUP="cm4-piControl-coupled"
EXISTING_RESULTS_DATASET="TODO"  # beaker dataset ID from coupled training (train.sh)
CKPT_TYPE="best_inference_ckpt"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_PATH#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=4

ATMOS_STATS_DATA=jamesd/2025-06-03-cm4-piControl-200yr-coupled-stats-atmosphere
OCEAN_STATS_DATA=jamesd/2025-06-03-cm4-piControl-200yr-coupled-stats-ocean

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

# --- Generate finetune-config.yaml from template + uncoupled configs ---

TEMPLATE_CONFIG_PATH="${SCRIPT_PATH}/finetune-config-template.yaml"
CONFIG_PATH="${SCRIPT_PATH}/finetune-config.yaml"

cp "${SCRIPT_PATH}/uncoupled-atmos/train-config.yaml" ./atmos-config.yaml
sed -i'' -e 's/statsdata/atmos_stats/g' ./atmos-config.yaml

cp "${SCRIPT_PATH}/uncoupled-ocean/train-config.yaml" ./ocean-config.yaml
sed -i'' -e 's/statsdata/ocean_stats/g' ./ocean-config.yaml

cp "$TEMPLATE_CONFIG_PATH" "$CONFIG_PATH"

# update component stepper configs, preserving template values on conflict
yq -i '.stepper.ocean.stepper *=n load("ocean-config.yaml").stepper' "$CONFIG_PATH"
yq -i '.stepper.atmosphere.stepper *=n load("atmos-config.yaml").stepper' "$CONFIG_PATH"

rm ./atmos-config.yaml ./ocean-config.yaml

# --- Validate and submit ---

python -m fme.coupled.validate_config "$CONFIG_PATH" --config_type train

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "Run SamudrACE CM4 piControl ocean + atmos fine-tuning" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --preemptible \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/jupiter-cirrascale \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $ATMOS_STATS_DATA:/atmos_stats \
    --dataset $OCEAN_STATS_DATA:/ocean_stats \
    --dataset "$EXISTING_RESULTS_DATASET:training_checkpoints/${CKPT_TYPE}.tar:/ckpt.tar" \
    --gpus $N_GPUS \
    --shared-memory 800GiB \
    --budget atec/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train "$CONFIG_PATH"
