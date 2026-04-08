#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-inference-oct-1978-2024"
JOB_GROUP="ace-aimip"
# this is from ace-aimip-fine-tune-decoder-pressure-levels-separate-decoder-lr-warmup-RS0
EXISTING_RESULTS_DATASET="01KAKXY0EK24K7BZK2N8SPJ5SJ"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

AIMIP_INFERENCE_CONFIG_FILENAME="ace-aimip-inference-config.yaml"
AIMIP_INFERENCE_BASE_CONFIG_PATH=$SCRIPT_PATH/$AIMIP_INFERENCE_CONFIG_FILENAME
AIMIP_INFERENCE_P2K_CONFIG_FILENAME="ace-aimip-inference-p2k-config.yaml"
AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH=$SCRIPT_PATH/$AIMIP_INFERENCE_P2K_CONFIG_FILENAME
AIMIP_INFERENCE_P4K_CONFIG_FILENAME="ace-aimip-inference-p4k-config.yaml"
AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH=$SCRIPT_PATH/$AIMIP_INFERENCE_P4K_CONFIG_FILENAME

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_CONFIG_PATH
python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH
python -m fme.ace.validate_config --config_type inference $AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH

launch_job () {

    local JOB_NAME=$1
    local TEMPLATE_CONFIG=$2
    local IC=$3
    local OVERRIDE=$4

    cd $REPO_ROOT && gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/ceres-cirrascale \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "sed 's/_r[0-9]i/_r${IC}i/g' ${TEMPLATE_CONFIG} > /tmp/ic-config.yaml && python -I -m fme.ace.inference /tmp/ic-config.yaml --override ${OVERRIDE}"

}

# launch 46-year (1979-2024) with spinup from 1978-10-01
# use 5 different initial conditions files
for IC in {1..5}; do
    JOB_NAME="${JOB_NAME_BASE}-IC${IC}"
    IC_PATH="/climate-default/2025-09-12-aimip-evaluation/aimip-evaluation-ics-v3/1978-09-30_IC$(( IC - 1 )).nc" # files are 0-indexed
    OUTPUT_PATH="/climate-default/2025-11-25-ace-aimip-inference-results/${JOB_NAME}"
    OVERRIDE="initial_condition.path=${IC_PATH} experiment_dir=${OUTPUT_PATH}"
    echo "Launching job $JOB_NAME with override: $OVERRIDE"
    launch_job "$JOB_NAME" "$AIMIP_INFERENCE_BASE_CONFIG_PATH" "$IC" "$OVERRIDE"
done

# same as above but use SST perturbed by +2K and +4K
for PERTURBATION in p2k p4k; do
    case $PERTURBATION in
        p2k) TEMPLATE_CONFIG=$AIMIP_INFERENCE_BASE_P2K_CONFIG_PATH ;;
        p4k) TEMPLATE_CONFIG=$AIMIP_INFERENCE_BASE_P4K_CONFIG_PATH ;;
    esac
    for IC in {1..5}; do
        JOB_NAME="${JOB_NAME_BASE}-${PERTURBATION}-IC${IC}"
        IC_PATH="/climate-default/2025-09-12-aimip-evaluation/aimip-evaluation-ics-v3/1978-09-30_IC$(( IC - 1 )).nc" # files are 0-indexed
        OUTPUT_PATH="/climate-default/2025-11-25-ace-aimip-inference-results/${JOB_NAME}"
        OVERRIDE="initial_condition.path=${IC_PATH} experiment_dir=${OUTPUT_PATH}"
        echo "Launching job: $JOB_NAME with perturbation: $PERTURBATION and IC: $IC"
        launch_job "$JOB_NAME" "$TEMPLATE_CONFIG" "$IC" "$OVERRIDE"
    done
done
