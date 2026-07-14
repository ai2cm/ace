#!/bin/bash

# Launches the three inference runs completing the matched AMIP CO2-attribution
# comparison (near-surface temperature invariance to CO2) over identical
# observed AMIP SST:
#   A: pre-FT donor checkpoint, rising (actual) CO2
#   B: pre-FT donor checkpoint, constant 1979 CO2
#   C: FT-1940 checkpoint,       rising (actual) CO2
# (The FT-1940 constant-CO2 run already exists; not relaunched here.)
#
# Optional first arg: a substring filter matched against JOB_NAME so a single
# arm can be (re)launched without touching the others. No arg launches all.

set -e

FILTER="${1:-}"

JOB_GROUP="mcgibbon-amip-co2-attribution"

# HARDCODED attribution: on this machine beaker whoami resolves to the jeremym
# service account, which misattributes. Beaker specs are immutable, so this
# must be mcgibbon, not $BEAKER_USERNAME.
WANDB_USERNAME="mcgibbon"

DONOR_CKPT="01KHJ5F1M6YKVZESPZAAVVD6G8"
FT1940_CKPT="01KQABCK4PPWSQ8X35XBBTQK35"

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

RISING_CONFIG_FILENAME="ace-inference-era5-amip.yaml"
CONSTANT_CONFIG_FILENAME="ace-inference-era5-constant-co2-amip.yaml"

python -m fme.ace.validate_config --config_type inference $RISING_CONFIG_FILENAME
python -m fme.ace.validate_config --config_type inference $CONSTANT_CONFIG_FILENAME

launch_job () {

    JOB_NAME=$1
    CKPT_DATASET=$2
    CONFIG_FILENAME=$3

    if [ -n "$FILTER" ] && [[ "$JOB_NAME" != *"$FILTER"* ]]; then
        echo "Skipping (filter '$FILTER'): $JOB_NAME"
        return
    fi

    CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

    echo "Launching job: $JOB_NAME  (ckpt $CKPT_DATASET, config $CONFIG_FILENAME)"

    cd $REPO_ROOT && gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'AMIP CO2-attribution inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/jupiter \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $CKPT_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --allow-dirty \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH

}

# A: pre-FT donor, rising (actual) CO2
launch_job "shield-donor-preFT-actual-co2-amip-sst-inference"   "$DONOR_CKPT"  "$RISING_CONFIG_FILENAME"

# B: pre-FT donor, constant 1979 CO2
launch_job "shield-donor-preFT-constant-1979-co2-amip-sst-inference" "$DONOR_CKPT" "$CONSTANT_CONFIG_FILENAME"

# C: FT-1940, rising (actual) CO2
launch_job "shield-ft-1940-2020-actual-co2-amip-sst-inference"  "$FT1940_CKPT" "$RISING_CONFIG_FILENAME"
