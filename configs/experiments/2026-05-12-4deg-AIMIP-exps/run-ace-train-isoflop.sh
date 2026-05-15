#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/run-ace-train.sh"

TAGS=(
    1.00x-D064   1.00x-D128   1.00x-D192   1.00x-D256   1.00x-D320   1.00x-D384
    0.50x-D064   0.50x-D128   0.50x-D192   0.50x-D256   0.50x-D320   0.50x-D384
    0.25x-D064   0.25x-D128   0.25x-D192   0.25x-D256   0.25x-D320   0.25x-D384
)

for tag in "${TAGS[@]}"; do
    bash "$TRAIN_SCRIPT" \
        "ace-train-config-4deg-AIMIP-${tag}.yaml" \
        "ace2-era5-train-4deg-AIMIP-isoflop-${tag}" \
        "ace2-era5-isoflop"
done
