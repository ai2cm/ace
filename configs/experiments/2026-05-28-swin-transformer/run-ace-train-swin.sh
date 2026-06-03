#!/opt/homebrew/bin/bash

set -e

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
JOB_GROUP="swin-transformer-comparison"
export WANDB_PROJECT=${WANDB_PROJECT:-"SwinTransformer"}
export BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-"ai2/climate-titan"}
export BEAKER_CLUSTER=${BEAKER_CLUSTER:-"ai2/titan"}
export BEAKER_PRIORITY=${BEAKER_PRIORITY:-"urgent"}

declare -A JOBS=(
  #["ace-train-config-4deg-AIMIP-swin-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-swin-v1-gmron"
  #["ace-train-config-4deg-AIMIP-swin-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-swin-v1-gmroff"
  ["ace-train-config-4deg-AIMIP-nc-swin-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-nc-swin-v1-gmron"
  #["ace-train-config-4deg-AIMIP-nc-swin-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-nc-swin-v1-gmroff"
)

for config in "${!JOBS[@]}"; do
  bash "$SCRIPT_DIR/run-ace-train.sh" "$config" "${JOBS[$config]}" "$JOB_GROUP"
done
