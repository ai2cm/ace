#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
BEAKER_USERNAME=troya
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

launch_job () {

    JOB_NAME=$1
    CHECKPOINT_DATASET=$2
    CONFIG_PATH=$3

    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

    SPIN_UP_OVERRIDES="experiment_dir=/results/spin-up \
        initial_condition.start_indices.times=[1989-01-01T00:00:00] \
        initial_condition.path=/climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0002/1989010100.nc \
        n_forward_steps=1460 \
        logging.log_to_wandb=false\
    "

    gantry run \
        --name $JOB_NAME \
        --description 'Run ACE inference for SHiELD model' \
        --task-name $JOB_NAME \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-ceres \
        --priority normal \
        --not-preemptible \
        --cluster ai2/saturn-cirrascale \
	--cluster ai2/ceres \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $CHECKPOINT_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --allow-dirty \
        --system-python \
        --install "pip install --no-deps ." \
        -- /bin/bash -c "\
            python -I -m fme.ace.evaluator $CONFIG_PATH \
            --override \
            n_ensemble_per_ic=25 \
            forward_steps_in_memory=2 \
          "

}

# checkpoint datasets
WEIGHTS=01K8MSK8RDWJB9GD7MEKAW2KEG


JOB_NAME="ace-inference-era5-pt-era5-multi-20-multi-ics-agg"

CONFIG_PATH="${SCRIPT_PATH}evaluate-xshield-amip-stochastic-ace.yaml"

launch_job $JOB_NAME $WEIGHTS $CONFIG_PATH
