#!/bin/bash

set -x

# directories for input data (training, validation, inference, stats)
export FME_TRAIN_DIR=/pscratch/sd/e/elynnwu/fme-dataset/2025-08-01-sample-E3SMv3-coupled-atm-wcycl1850-r025.zarr
export FME_VALID_DIR=/pscratch/sd/e/elynnwu/fme-dataset/2025-08-01-sample-E3SMv3-coupled-atm-wcycl1850-r025.zarr
export FME_STATS_DIR=/pscratch/sd/r/rebassoo/fme-preprocess/2025-04-01-e3smv3-1deg/2025-04-01-e3smv3-1deg
export EMBED_DIM_VALUE=256
export SCALE_FACTOR_VALUE=1
export H_PARALLEL_SIZE=8
export W_PARALLEL_SIZE=8

nodes=16
# wandb config
export WANDB_NAME=PM-EAMv3-wcycl1850-25km-old-branch-${nodes}nodes-sp-${H_PARALLEL_SIZE}x${W_PARALLEL_SIZE}-n${EMBED_DIM_VALUE}-scale-factor-${SCALE_FACTOR_VALUE}-train
export WANDB_RUN_GROUP=wcycl1850-25km

export COMMIT=$(git rev-parse --short HEAD)

# if resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job
# export RESUME_JOB_ID=45451769

# user should not need to modify below

# copy config to staging area so that local changes between job submission
# and job start will not effect the run
UUID=$(uuidgen)
export CONFIG_DIR=${PSCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
if [ -z "${RESUME_JOB_ID}" ]; then
  cp config-finetune.yaml $CONFIG_DIR/train-config.yaml
else
  cp ${PSCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml $CONFIG_DIR/train-config.yaml
fi
cp run-train-perlmutter.sh $CONFIG_DIR/run-train-perlmutter.sh  # copy for reproducibility/tracking
cp sbatch-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh
cp make-venv.sh $CONFIG_DIR/make-venv.sh
cp upload-to-beaker.sh $CONFIG_DIR/upload-to-beaker.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
conda activate $FME_VENV
python -m fme.ace.validate_config --config_type train $CONFIG_DIR/train-config.yaml
#sbatch -t 00:10:00 -q debug sbatch-scripts/sbatch-train.sh  # use this for debugging config/submission
sbatch sbatch-scripts/sbatch-train.sh
