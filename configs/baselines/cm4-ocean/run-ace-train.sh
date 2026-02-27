#!/bin/bash

set -e

CONFIG_FILENAME="ace-train-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4
PRIORITY="high"
WORKSPACE="ai2/ace"

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

JOB_GROUP="2025-08-28-baseline" # update when training a new baseline
RS=1
JOB_NAME="${JOB_GROUP}-rs${RS}-train"
echo "Job name: ${JOB_NAME}"

OVERRIDE_ARGS= # add group-specific overrides here, e.g. lr, max_epochs, etc.
STATS_DATA=jamesd/2025-08-22-cm4-piControl-200yr-coupled-stats-ocean

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

EXPERIMENT_ID=$(
    gantry run $ALLOW_DIRTY \
      --name $JOB_NAME \
      --description "ACE-Saumdra CM4 baseline training RS${RS}" \
      --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
      --workspace $WORKSPACE \
      --priority $PRIORITY \
      --preemptible \
      --cluster ai2/ceres \
      --cluster ai2/jupiter \
      --cluster ai2/neptune \
      --cluster ai2/saturn \
      --env WANDB_USERNAME=$BEAKER_USERNAME \
      --env WANDB_NAME=$JOB_NAME \
      --env WANDB_JOB_TYPE=training \
      --env WANDB_RUN_GROUP=$JOB_GROUP \
      --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset-secret google-credentials:/tmp/google_application_credentials.json \
      --dataset $STATS_DATA:/statsdata \
      --gpus $N_GPUS \
      --shared-memory 400GiB \
      --weka climate-default:/climate-default \
      --budget ai2/climate \
      --system-python \
      --install "pip install --no-deps ." \
      -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH --override $OVERRIDE_ARGS |
      tee /dev/tty |
      grep beaker.org |
      cut -d/ -f5
)
# remove or change 'training' once completed in order to submit an evaluator job
echo "${JOB_GROUP}|${RS}|${EXPERIMENT_ID}|training|best_inference_ckpt" >> $SCRIPT_PATH/experiments.txt
echo
sleep 1
