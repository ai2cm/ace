#!/bin/bash

set -x

export COMMIT=$(git rev-parse --short HEAD)

# wandb config
#export WANDB_NAME=changeme
#export WANDB_RUN_GROUP=changeme

# ALCF project / scratch locations
export PROJECT=E3SMinput
export FME_SCRATCH=/eagle/E3SMinput/elynnwu/scratch

# PBS resource request
export QUEUE=preemptable
export WALLTIME=01:00:00
export FILESYSTEMS=home:eagle

# directory for trained model checkpoint and validation data (update these)
export FME_CHECKPOINT_PATH=/eagle/E3SMinput/elynnwu/scratch/fme-output/1234567/training_checkpoints/best_inference_ckpt.tar
export FME_VALID_DIR=/eagle/E3SMinput/elynnwu/fme-dataset

# ---- user should not need to modify below ----

# copy config to staging area so that local changes between job submission
# and job start will not affect the run
UUID=$(uuidgen)
export CONFIG_DIR=${FME_SCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
cp config-inference.yaml $CONFIG_DIR/config-inference.yaml
cp make-venv.sh $CONFIG_DIR/make-venv.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)

mkdir -p joblogs

qsub -V \
  -A ${PROJECT} \
  -q ${QUEUE} \
  -l select=1:system=polaris \
  -l place=scatter \
  -l filesystems=${FILESYSTEMS} \
  -l walltime=${WALLTIME} \
  pbs-scripts/pbs-inference.sh
