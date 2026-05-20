#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/run-ace-train.sh"

TAGS=(
    variable-masking-0.80
)

for tag in "${TAGS[@]}"; do
    bash "$TRAIN_SCRIPT" \
        "ace-train-config-4deg-AIMIP-${tag}.yaml" \
        "ace2-era5-train-4deg-${tag}" \
        "ace2-era5-masked"
done
