#!/bin/bash
# Launch the 10-year segmented inference for TC tracking (ACE2S-6h model).
# Run FROM configs/baselines/era5-ace2s-ablations/.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH=$(git rev-parse --show-prefix)
cd "$REPO_ROOT"

CONFIG_PATH="${SCRIPT_PATH}ace2s-inference-6h-tc.yaml"

# The 6h fine-tune result dataset (exit 0 job of 1deg-6h-ace2s-ft3-detached-rs0)
FINETUNE_RESULT="01M3T8A93QD5T2YYJFBQ4ATJT2"

python -m fme.ace.validate_config --config_type inference "$CONFIG_PATH"

gantry run \
  --name "1deg-6h-ace2s-tc-inference-10yr" \
  --description 'ACE2S-6h 10yr segmented inference for TC tracking' \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority normal \
  --not-preemptible \
  --cluster ai2/jupiter \
  --env WANDB_USERNAME=mcgibbon \
  --env WANDB_NAME="1deg-6h-ace2s-tc-inference-10yr" \
  --env WANDB_JOB_TYPE=inference \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --gpus 1 \
  --shared-memory 50GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --allow-dirty \
  --system-python \
  --install "pip install --no-deps ." \
  --dataset "${FINETUNE_RESULT}:/weights" \
  -- python -m fme.ace.inference "$CONFIG_PATH" \
     --segments 10 \
     --override checkpoint_path=/weights/training_checkpoints/best_inference_ckpt.tar
