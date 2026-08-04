#!/bin/bash

set -x

# wandb config
export WANDB_NAME=PL-v20260602-Samudra-piControl-105yr-rs0
export WANDB_RUN_GROUP=v20260602

export COMMIT=$(git rev-parse --short HEAD)

# ALCF project / scratch locations
export PROJECT=E3SMinput
export FME_SCRATCH=/eagle/E3SMinput/elynnwu

# PBS resource request (preemptable queue: 1-10 nodes, long walltime, may be
# preempted -> job is automatically requeued because pbs-train.sh sets -r y).
export QUEUE=preemptable
export NUM_NODES=2
export WALLTIME=12:00:00
export FILESYSTEMS=home:eagle

# Seconds before WALLTIME at which the job checkpoints and auto-resubmits itself
# to continue from the latest checkpoint (handles walltime expiry, since PBS
# does not auto-requeue on timeout). Increase if a clean stop needs more time.
export SOFT_LIMIT_MARGIN=300

# directories for input data (update these to your Polaris/eagle paths)
export FME_TRAIN_DIR=/eagle/E3SMinput/elynnwu/fme-dataset
export FME_VALID_DIR=/eagle/E3SMinput/elynnwu/fme-dataset
export FME_STATS_DIR=/eagle/E3SMinput/elynnwu/fme-dataset/2026-06-02-E3SMv3-piControl-105yr-coupled-stats/ocean

# if resuming a failed/preempted job, provide its PBS job id (numeric part) below
# and uncomment; keep the settings above consistent with the original job
export RESUME_JOB_ID=7184494

# ---- user should not need to modify below ----

# copy config to staging area so that local changes between job submission
# and job start will not affect the run
UUID=$(uuidgen)
export CONFIG_DIR=${FME_SCRATCH}/fme-config/${UUID}
mkdir -p $CONFIG_DIR
if [ -z "${RESUME_JOB_ID}" ]; then
  cp config-train.yaml $CONFIG_DIR/train-config.yaml
else
  cp ${FME_SCRATCH}/fme-output/${RESUME_JOB_ID}/job_config/train-config.yaml $CONFIG_DIR/train-config.yaml
fi
cp run-train-polaris.sh $CONFIG_DIR/run-train-polaris.sh  # copy for reproducibility/tracking
cp pbs-scripts/requeueable-train.sh $CONFIG_DIR/requeueable-train.sh
cp pbs-scripts/set-affinity.sh $CONFIG_DIR/set-affinity.sh
cp make-venv.sh $CONFIG_DIR/make-venv.sh
cp upload-to-beaker.sh $CONFIG_DIR/upload-to-beaker.sh

export FME_VENV=$($CONFIG_DIR/make-venv.sh $COMMIT | tail -n 1)
module use /soft/modulefiles
module load conda
conda activate $FME_VENV
set -e
# python -m fme.ace.validate_config --config_type train $CONFIG_DIR/train-config.yaml

mkdir -p joblogs

# Submit the PBS job. "-V" forwards the current environment (CONFIG_DIR, FME_VENV,
# data dirs, wandb config, RESUME_JOB_ID, FME_SCRATCH, ...) to the batch job.
#
# To quickly test config/submission, use the debug queue with a short walltime
# (max 2 nodes, max 1 hr):
# qsub -V -A ${PROJECT} -q debug -l select=2:system=polaris -l place=scatter \
#       -l filesystems=${FILESYSTEMS} -l walltime=00:30:00 pbs-scripts/pbs-train.sh
qsub -V \
  -A ${PROJECT} \
  -q ${QUEUE} \
  -l select=${NUM_NODES}:system=polaris \
  -l place=scatter \
  -l filesystems=${FILESYSTEMS} \
  -l walltime=${WALLTIME} \
  pbs-scripts/pbs-train.sh
