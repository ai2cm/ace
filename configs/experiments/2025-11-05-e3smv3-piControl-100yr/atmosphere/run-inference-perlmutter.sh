#!/bin/bash

set -x

export COMMIT=06e96385f

# wandb config
export WANDB_NAME=PM-v1103-Samudra-piControl-100yr-unstable-bottom-layer-eval
export WANDB_RUN_GROUP=v1103-filter-scale-4

# directory for trained model and validation data
export FME_CHECKPOINT_PATH=/pscratch/sd/e/elynnwu/fme-output/44949721/training_checkpoints/ckpt.tar
export FME_VALID_DIR=/pscratch/sd/e/elynnwu/fme-dataset

# copy config to staging area so that local changes between job submission
# and job start will not effect the run
UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
cp config-inference.yaml $CONFIG_DIR/config-inference.yaml
cp make-venv.sh $CONFIG_DIR/make-venv.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)

sbatch -t 00:30:00 -q debug sbatch-scripts/sbatch-inference.sh # use this for debugging config/submission
# sbatch -t 03:00:00 sbatch-scripts/sbatch-inference.sh
