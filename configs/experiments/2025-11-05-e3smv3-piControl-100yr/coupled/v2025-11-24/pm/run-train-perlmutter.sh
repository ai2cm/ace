#!/bin/bash

set -x

# wandb config
export WANDB_NAME=PM-fto-v1124-train-95yr-ebpeysbb-dso4ysdg-ft-AMIP-train
export WANDB_RUN_GROUP=fto

export COMMIT=$(git rev-parse --short HEAD)
export OCEAN_CKPT=/pscratch/sd/e/elynnwu/fme-output/48937597/training_checkpoints/best_inference_ckpt.tar
export ATMOS_CKPT=/pscratch/sd/e/elynnwu/fme-output/ebpeysbb//training_checkpoints/best_inference_ckpt.tar

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
python -m fme.coupled.validate_config --config_type train $CONFIG_DIR/train-config.yaml
sbatch sbatch-scripts/sbatch-train.sh
