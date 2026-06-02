#!/opt/homebrew/bin/bash

set -e

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
JOB_GROUP="swin-transformer-comparison"
export WANDB_PROJECT=${WANDB_PROJECT:-SwinTransformer}
export BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/ace}
export BEAKER_CLUSTER=${BEAKER_CLUSTER:-ai2/titan}
export BEAKER_PRIORITY=${BEAKER_PRIORITY:-high}

declare -A JOBS=(
  #["ace-train-config-4deg-AIMIP-swin-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-swin-v2-earth-pad-gmron"
  ["ace-train-config-4deg-AIMIP-swin-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-swin-v2-earth-pad-gmroff"
  #["ace-train-config-4deg-AIMIP-nc-swin-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-nc-swin-v2-earth-pad-gmron"
  ["ace-train-config-4deg-AIMIP-nc-swin-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-nc-swin-v2-earth-pad-gmroff"
  ["ace-train-config-4deg-AIMIP-sfno-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-sfno-gmron"
  ["ace-train-config-4deg-AIMIP-sfno-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-sfno-gmroff"
  #["ace-train-config-4deg-AIMIP-nc-sfno-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-nc-sfno-gmron"
  ["ace-train-config-4deg-AIMIP-nc-sfno-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-nc-sfno-gmroff"
  #["ace-train-config-4deg-AIMIP-crossformer-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-crossformer-gmron"
  ["ace-train-config-4deg-AIMIP-crossformer-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-crossformer-gmroff"
  #["ace-train-config-4deg-AIMIP-nc-crossformer-gmron.yaml"]="ace2-era5-train-4deg-AIMIP-nc-crossformer-gmron"
  ["ace-train-config-4deg-AIMIP-nc-crossformer-gmroff.yaml"]="ace2-era5-train-4deg-AIMIP-nc-crossformer-gmroff"
)

for config in "${!JOBS[@]}"; do
  bash "$SCRIPT_DIR/run-ace-train.sh" "$config" "${JOBS[$config]}" "$JOB_GROUP"
done
