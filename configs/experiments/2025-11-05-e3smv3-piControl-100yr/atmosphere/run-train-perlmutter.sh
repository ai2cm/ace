#!/bin/bash

set -x

# wandb config
export WANDB_NAME=PM-v1124-ACE-E3SMv3-piControl-100yr-energy-cs-train-rs1
export WANDB_RUN_GROUP=v1124-energy-cs

export COMMIT=$(git rev-parse --short HEAD)

export FME_TRAIN_DIR=/pscratch/sd/e/elynnwu/fme-dataset
export FME_STATS_DIR=/pscratch/sd/e/elynnwu/fme-dataset/2025-11-24-E3SMv3-piControl-100yr-coupled-stats/uncoupled_atmosphere

# if resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job
# export RESUME_JOB_ID=45114004

# user should not need to modify below

# copy config to staging area so that local changes between job submission
# and job start will not effect the run
UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
if [ -z "${RESUME_JOB_ID}" ]; then
  cp config-train.yaml $CONFIG_DIR/train-config.yaml
else
  cp ${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml $CONFIG_DIR/train-config.yaml
fi
cp run-train-perlmutter.sh $CONFIG_DIR/run-train-perlmutter.sh  # copy for reproducibility/tracking
cp sbatch-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh
cp make-venv.sh $CONFIG_DIR/make-venv.sh
cp upload-to-beaker.sh $CONFIG_DIR/upload-to-beaker.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
conda activate $FME_VENV
set -e
python -m fme.ace.validate_config --config_type train $CONFIG_DIR/train-config.yaml
sbatch sbatch-scripts/sbatch-train.sh
