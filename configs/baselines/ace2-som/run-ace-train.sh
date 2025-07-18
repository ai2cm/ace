#!/bin/bash

set -e

JOB_NAME="ace-train"  # recommended but not required to change this
CONFIG_FILENAME="ace-train-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8
STATS_DATASET=andrep/2025-03-14-vertically-resolved-1deg-fme-c96-shield-som-increasing-co2-dataset-stats

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

PRE_TRAIN_EXPERIMENT_DIR=/results/pre-train-without-multi-call
PRE_TRAIN_OVERRIDE="\
    experiment_dir=$PRE_TRAIN_EXPERIMENT_DIR \
    max_epochs=50 \
"
FINE_TUNE_EXPERIMENT_DIR=/results/fine-tune-with-multi-call
FINE_TUNE_OVERRIDE="\
    experiment_dir=$FINE_TUNE_EXPERIMENT_DIR \
    max_epochs=10 \
    stepper.step.config.config.forcing_name=global_mean_co2 \
    stepper.step.config.config.forcing_multipliers._with_quartered_co2=0.25 \
    stepper.step.config.config.forcing_multipliers._with_quadrupled_co2=4.0 \
    stepper.step.config.config.output_names=[ULWRFsfc,ULWRFtoa,DLWRFsfc,DSWRFsfc,USWRFsfc,USWRFtoa] \
    stepper.step.config.include_multi_call_in_loss=true \
    stepper.parameter_init.weights_path=$PRE_TRAIN_EXPERIMENT_DIR/training_checkpoints/best_inference_ckpt.tar \
"

python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $PRE_TRAIN_OVERRIDE
python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $FINE_TUNE_OVERRIDE
gantry run \
    --name $JOB_NAME \
    --description 'Run ACE training' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --preemptible \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/ceres-cirrascale \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $STATS_DATASET:/statsdata \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-conda \
    --install "pip install --no-deps ." \
    -- bash -c "\
       WANDB_NAME=$JOB_NAME-pre-train torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $PRE_TRAIN_OVERRIDE \
       && \
       WANDB_NAME=$JOB_NAME-fine-tune torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $FINE_TUNE_OVERRIDE \
    "
