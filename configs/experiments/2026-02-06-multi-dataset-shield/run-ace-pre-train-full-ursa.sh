#!/bin/bash

set -e

# https://askubuntu.com/questions/893911/when-writing-a-bash-script-how-do-i-get-the-absolute-path-of-the-location-of-th
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

COMMIT=3d05b879ffb9a8208a4db845e140db2f3bde376e
URSA_CONDA_DIR=$HOME/software/vcm-workflow-control/examples/ace/ursa-conda
TRAIN_SUBMISSION_SCRIPT=$URSA_CONDA_DIR/run-train-ursa.sh
SCRATCH=/scratch4/GFDL/gfdlhires/$USER
FME_VENV=$($URSA_CONDA_DIR/make-venv.sh $COMMIT $SCRATCH/fme-env $SCRATCH | tail -n 1)

# If resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job.
# export RESUME_JOB_ID=12345678

CONFIG_FILENAME="one-step-pre-train-config-full-ursa.yaml"
CONFIG_PATH=$SCRIPT_DIR/$CONFIG_FILENAME
WANDB_USERNAME=spencerc_ai2
export WANDB_RUN_GROUP=ace-shield
for seed in 0 1 2 3
do
    override="seed=${seed}"
    wandb_name=ace-shield-one-step-pre-train-full-rs${seed}
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
done
