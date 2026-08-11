#!/bin/bash

set -e

# https://askubuntu.com/questions/893911/when-writing-a-bash-script-how-do-i-get-the-absolute-path-of-the-location-of-th
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

COMMIT=cc5db6a1067d6caa1da4375b772d1f8886bc8dfa  # With all of Oli's optimizations
URSA_CONDA_DIR=$HOME/software/vcm-workflow-control/examples/ace/ursa-conda
TRAIN_SUBMISSION_SCRIPT=$URSA_CONDA_DIR/run-train-ursa.sh
SCRATCH=/scratch4/GFDL/gfdlhires/$USER
FME_VENV=$($URSA_CONDA_DIR/make-venv.sh $COMMIT $SCRATCH/fme-env $SCRATCH | tail -n 1)

# If resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job.
# export RESUME_JOB_ID=12345678

# Constant environment variables
WANDB_USERNAME=spencerc_ai2
export WANDB_RUN_GROUP=ace-shield-profile
SEED=0
override="seed=${SEED}"

CONFIG_FILENAME="one-step-pre-train-zarr.yaml"
CONFIG_PATH=$SCRIPT_DIR/$CONFIG_FILENAME
wandb_name=ace-shield-one-step-pre-train-zarr-${COMMIT}-rs${SEED}
conda run --prefix $FME_VENV \
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override
$TRAIN_SUBMISSION_SCRIPT \
    $FME_VENV \
    $CONFIG_PATH \
    $URSA_CONDA_DIR \
    $SCRATCH \
    $wandb_name \
    $WANDB_USERNAME \
    $override

CONFIG_FILENAME="one-step-pre-train-netcdf.yaml"
CONFIG_PATH=$SCRIPT_DIR/$CONFIG_FILENAME
wandb_name=ace-shield-one-step-pre-train-netcdf-${COMMIT}-rs${SEED}
conda run --prefix $FME_VENV \
    python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $override
$TRAIN_SUBMISSION_SCRIPT \
    $FME_VENV \
    $CONFIG_PATH \
    $URSA_CONDA_DIR \
    $SCRATCH \
    $wandb_name \
    $WANDB_USERNAME \
    $override
