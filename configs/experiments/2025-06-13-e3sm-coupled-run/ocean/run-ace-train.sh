#!/bin/bash

set -e

CONFIG_FILENAME="train-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=elynnwu
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=8

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

JOB_GROUP="Samudra-E3SM-with-atmos-5daily-sfc-fluxes-oic-masked" # should be the same as the evaluator run job group
JOB_STEM="${JOB_GROUP}-train"  # update when training a new baseline

GROUP_OVERRIDE_ARGS= # add group-specific overrides here, e.g. lr, max_epochs, etc.
STATS_DATA=elynn/2025-06-13-E3SM-piControl-ocean-with-atmos-sfc-var-stats

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

N_RANDOM_SEED_RUNS=4

for RS in $(seq 1 $N_RANDOM_SEED_RUNS); do
    JOB_NAME="${JOB_STEM}-rs${RS}"  # job name for the current random seed
    # don't log validation maps
    OVERRIDE_ARGS="${GROUP_OVERRIDE_ARGS} validation_aggregator.log_snapshots=false validation_aggregator.log_mean_maps=false"
    if [ $RS -gt 1 ]; then
        PRIORITY="high"
        ALLOW_DIRTY=--allow-dirty # needed since experiments.txt will be updated
    else
        PRIORITY="normal"
        ALLOW_DIRTY=
    fi
    if [[ -n "${OVERRIDE_ARGS}" ]]; then
      OVERRIDE="--override ${OVERRIDE_ARGS}"
    else
      OVERRIDE=
    fi
    echo "Job name: ${JOB_NAME}"
    EXPERIMENT_ID=$(
        gantry run $ALLOW_DIRTY \
          --name $JOB_NAME \
          --description "Saumdra E3SM training RS${RS}" \
          --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
          --workspace ai2/ace \
          --priority $PRIORITY \
          --preemptible \
          --cluster ai2/jupiter-cirrascale-2 \
          --cluster ai2/ceres-cirrascale \
          --weka climate-default:/climate-default \
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
          --budget ai2/climate \
          --no-conda \
          --install "pip install --no-deps ." \
          -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH $OVERRIDE |
          tee /dev/tty |
          grep beaker.org |
          cut -d/ -f5
    )
    # remove or change 'training' once completed in order to submit an evaluator job
    echo "${JOB_GROUP}|${RS}|${EXPERIMENT_ID}|training|best_inference_ckpt" >> $SCRIPT_PATH/experiments.txt
    echo "${JOB_GROUP}|${RS}|${EXPERIMENT_ID}|training|best_ckpt" >> $SCRIPT_PATH/experiments.txt
    echo "${JOB_GROUP}|${RS}|${EXPERIMENT_ID}|training|ckpt" >> $SCRIPT_PATH/experiments.txt
    echo
    sleep 1
done
