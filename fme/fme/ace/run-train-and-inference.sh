#!/bin/bash

set -e

SCRIPT_DIRECTORY=${0%/*}

YAML_TRAIN_CONFIG=$1
YAML_INFERENCE_CONFIG=$2
NPROC_PER_NODE=$3

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $NPROC_PER_NODE $SCRIPT_DIRECTORY/train.py --yaml_config $YAML_TRAIN_CONFIG

echo ===============================================================================
echo ==================== FINISHED TRAINING / STARTING INFERENCE ===================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
python -m fme.ace.inference $YAML_INFERENCE_CONFIG
