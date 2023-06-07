#!/bin/bash

set -e

SCRIPT_DIRECTORY=${0%/*}

YAML_CONFIG=$1
NPROC_PER_NODE=$2

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $NPROC_PER_NODE $SCRIPT_DIRECTORY/train.py --yaml_config $YAML_CONFIG

echo ===============================================================================
echo ==================== FINISHED TRAINING / STARTING INFERENCE ===================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
python $SCRIPT_DIRECTORY/inference/inference.py --yaml_config $YAML_CONFIG --vis
