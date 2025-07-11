#!/bin/bash

set -e

YAML_TRAIN_CONFIG=$1
YAML_INFERENCE_CONFIG=$2
NPROC_PER_NODE=$3

# run training
export WANDB_JOB_TYPE=training
torchrun --nproc_per_node $NPROC_PER_NODE -m fme.ace.train $YAML_TRAIN_CONFIG

echo ===============================================================================
echo ==================== FINISHED TRAINING / STARTING INFERENCE ===================
echo ===============================================================================

# run inference
export WANDB_JOB_TYPE=inference
python -m fme.ace.inference.evaluator $YAML_INFERENCE_CONFIG
