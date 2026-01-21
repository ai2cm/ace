#!/bin/bash

set -e

JOB_NAME="ace-aimip-train-multi-stage-with-co2-rs0"
JOB_GROUP="ace21-era5"
CONFIG_FILENAME="ace-train-config-with-co2.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

PRE_TRAIN_EXPERIMENT_DIR=/results/pre-train-without-ec
PRE_TRAIN_OVERRIDE="\
    experiment_dir=$PRE_TRAIN_EXPERIMENT_DIR \
    max_epochs=120 \
"

FINE_TUNE_EXPERIMENT_DIR=/results/fine-tune-with-ec
FINE_TUNE_OVERRIDE="\
    experiment_dir=$FINE_TUNE_EXPERIMENT_DIR \
    max_epochs=25 \
    stepper.step.config.corrector.total_energy_budget_correction.method=constant_temperature \
    stepper.step.config.corrector.total_energy_budget_correction.constant_unaccounted_heating=6.62 \
    stepper.parameter_init.weights_path=$PRE_TRAIN_EXPERIMENT_DIR/training_checkpoints/best_inference_ckpt.tar \
"

python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $PRE_TRAIN_OVERRIDE
python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $FINE_TUNE_OVERRIDE

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description 'Run ACE multi-stage training: 120 epochs without EC, then 25 epochs with EC' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority high \
    --preemptible \
    --cluster ai2/ceres \
    --cluster ai2/jupiter \
    --env WANDB_USERNAME=$WANDB_USERNAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset oliverwm/era5-1deg-8layer-stats-1990-2019-v2:/statsdata \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c "\
       WANDB_NAME=$JOB_NAME-pre-train torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $PRE_TRAIN_OVERRIDE \
       && \
       WANDB_NAME=$JOB_NAME-fine-tune torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $FINE_TUNE_OVERRIDE \
    "
