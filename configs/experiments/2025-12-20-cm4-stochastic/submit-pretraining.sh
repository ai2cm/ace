#!/bin/bash

set -e

SCRATCH=/scratch4/GFDL/gfdlscr/$USER
WANDB_USERNAME=spencerc_ai2
COMMIT=ba6cc3709ddd98ef2d1ea278804e0009f8c70276

# https://askubuntu.com/questions/893911/when-writing-a-bash-script-how-do-i-get-the-absolute-path-of-the-location-of-th
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

URSA_CONDA_DIR=$HOME/software/vcm-workflow-control-repo-switch/examples/ace/ursa-conda
TRAIN_SUBMISSION_SCRIPT=$URSA_CONDA_DIR/run-train-ursa.sh
FME_VENV=$($URSA_CONDA_DIR/make-venv.sh $COMMIT $SCRATCH/fme-env | tail -n 1)

# If resuming a failed job, provide its slurm job ID below and uncomment;
# note that information entered above should be consistent with that of
# the failed job.
# export RESUME_JOB_ID=12345678

# CONFIG_FILENAME="ursa-stochastic-pretrain.yaml"
# CONFIG_PATH=$SCRIPT_DIR/$CONFIG_FILENAME
# export WANDB_RUN_GROUP=cm4_like_am4_random_co2_stochastic_pretrain
# for name in cm4_like_am4_random_co2_stochastic_pretrain-rs0-train cm4_like_am4_random_co2_stochastic_pretrain-rs1-train
# do
#     OVERRIDE=
#     WANDB_NAME=$name
#     conda run --prefix $FME_VENV \
# 	  python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $OVERRIDE
#     $TRAIN_SUBMISSION_SCRIPT \
# 	$FME_VENV \
# 	$CONFIG_PATH \
# 	$URSA_CONDA_DIR \
# 	$SCRATCH \
# 	$WANDB_NAME \
# 	$WANDB_USERNAME \
# 	$OVERRIDE
# done

CONFIG_FILENAME="ursa-stochastic-pretrain-limited.yaml"
CONFIG_PATH=$SCRIPT_DIR/$CONFIG_FILENAME
export WANDB_RUN_GROUP=cm4_like_am4_random_co2_limited_stochastic_pretrain
for name in cm4_like_am4_random_co2_limited_stochastic-rs0-pretrain cm4_like_am4_random_co2_limited_stochastic-rs1-pretrain
do
    OVERRIDE=
    WANDB_NAME=$name
    conda run --prefix $FME_VENV \
	  python -m fme.ace.validate_config --config_type train $CONFIG_PATH --override $OVERRIDE
    $TRAIN_SUBMISSION_SCRIPT \
	$FME_VENV \
	$CONFIG_PATH \
	$URSA_CONDA_DIR \
	$SCRATCH \
	$WANDB_NAME \
	$WANDB_USERNAME \
	$OVERRIDE
done
