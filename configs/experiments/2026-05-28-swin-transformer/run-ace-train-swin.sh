#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
JOB_GROUP="2026-05-28-swin-transformer"
export WANDB_PROJECT=${WANDB_PROJECT:-swin-transformer}
export BEAKER_WORKSPACE=${BEAKER_WORKSPACE:-ai2/climate-titan}
export BEAKER_CLUSTER=${BEAKER_CLUSTER:-ai2/titan}

declare -A JOBS=(
  ["ace-train-config-4deg-AIMIP-nc-swin.yaml"]="ace2-era5-train-4deg-AIMIP-nc-swin"
  ["ace-train-config-4deg-AIMIP-sfno.yaml"]="ace2-era5-train-4deg-AIMIP-sfno"
  ["ace-train-config-4deg-AIMIP-swin.yaml"]="ace2-era5-train-4deg-AIMIP-swin"
)

for config in "${!JOBS[@]}"; do
  bash "$SCRIPT_DIR/run-ace-train.sh" "$config" "${JOBS[$config]}" "$JOB_GROUP"
done
