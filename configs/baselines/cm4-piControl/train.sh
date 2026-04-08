#!/bin/bash
#
# SamudrACE CM4 piControl training stage 1: freezes the atmosphere model and
# fine-tunes the ocean model in coupled mode.

set -e

JOB_NAME="cm4-piControl-coupled-train"
JOB_GROUP="cm4-piControl-coupled"
EXISTING_RESULTS_ATMOS_DATASET="01JXXESTVASYBEKBM1VAWCRV87"  # beaker dataset ID from uncoupled atmos training
EXISTING_RESULTS_OCEAN_DATASET="01JX4DEKY2A13D6Y95T53DSVCQ"  # beaker dataset ID from uncoupled ocean training
ATMOS_CKPT="best_inference_ckpt"
OCEAN_CKPT="best_inference_ckpt"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_PATH#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=4

ATMOS_STATS_DATA=jamesd/2025-06-03-cm4-piControl-200yr-coupled-stats-atmosphere
OCEAN_STATS_DATA=jamesd/2025-06-03-cm4-piControl-200yr-coupled-stats-ocean

cd "$REPO_ROOT"  # so config path is valid no matter where we are running this script

# --- Generate train-config.yaml from template + uncoupled configs ---

TEMPLATE_CONFIG_PATH="${SCRIPT_PATH}/train-config-template.yaml"
CONFIG_PATH="${SCRIPT_PATH}/train-config.yaml"

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
    --description "Run SamudrACE CM4 piControl ocean-only fine-tuning" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --preemptible \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/saturn-cirrascale \
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
    --dataset "$EXISTING_RESULTS_ATMOS_DATASET:training_checkpoints/${ATMOS_CKPT}.tar:/atmos_ckpt.tar" \
    --dataset "$EXISTING_RESULTS_OCEAN_DATASET:training_checkpoints/${OCEAN_CKPT}.tar:/ocean_ckpt.tar" \
    --gpus $N_GPUS \
    --shared-memory 800GiB \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train "$CONFIG_PATH"
