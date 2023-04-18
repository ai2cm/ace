#!/bin/bash

YAML_CONFIG=$1
CONFIG_SETTING=$2
NPROC_PER_NODE=$3

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $NPROC_PER_NODE /full-model/fme/fme/fcn_training/train.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING

echo ===============================================================================
echo ==================== FINISHED TRAINING / STARTING INFERENCE ===================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
python /full-model/fme/fme/fcn_training/inference/inference.py --yaml_config $YAML_CONFIG --config $CONFIG_SETTING --vis
