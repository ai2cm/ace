#!/bin/bash

set -e

SCRATCH=/scratch4/GFDL/gfdlhires/$USER
WANDB_USERNAME=spencerc_ai2
COMMIT=43de3254a21bfe2727e5232ff1f8484d66bf7dbc

# https://askubuntu.com/questions/893911/when-writing-a-bash-script-how-do-i-get-the-absolute-path-of-the-location-of-th
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

URSA_CONDA_DIR=$HOME/software/vcm-workflow-control/examples/ace/ursa-conda
TRAIN_SUBMISSION_SCRIPT=$URSA_CONDA_DIR/run-train-ursa.sh
FME_VENV=$($URSA_CONDA_DIR/make-venv.sh $COMMIT $SCRATCH/fme-env $SCRATCH | tail -n 1)
CHECKPOINT_ROOT="/home/Spencer.Clark/scratch/fme-output"

# If resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job.
# export RESUME_JOB_ID=12345678

declare -A weights_ids=( [0]="6549001" [1]="6549002" )
CONFIG_FILENAME="ursa-stochastic-train.yaml"
CONFIG_PATH=$SCRIPT_DIR/$CONFIG_FILENAME
CHECKPOINT="best_ckpt.tar"
export WANDB_RUN_GROUP=cm4_like_am4_random_co2_stochastic_ec_train
for seed in 0 1
do
    weights_id=${weights_ids[$seed]}
    weights=${CHECKPOINT_ROOT}/${weights_id}/training_checkpoints/${CHECKPOINT}
    override="stepper_training.parameter_init.weights_path=$weights"
    wandb_name=cm4_like_am4_random_co2_stochastic_ec-rs${seed}-train
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
