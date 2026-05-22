#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$0")

EXISTING_RESULTS_DATASETS=(
    "01KS3R14YEQRRGGTPST0213AGJ"
    "01KS3R0YARDEGB6XVAX86R1CJE"
    "01KS3X8VJW4VP580XVDMXKPMHC"
)

DATASET_NAMES=(
    "var-masking-0.80-300e"
    "var-masking-0.20-300e"
    "var-masking-0.80"
)

for i in "${!EXISTING_RESULTS_DATASETS[@]}"; do
    dataset="${EXISTING_RESULTS_DATASETS[$i]}"
    name="${DATASET_NAMES[$i]}"
    "$SCRIPT_DIR/run-ace-eval.sh" "ace-eval-config-4deg-AIMIP.yaml" "ace-eval-4deg-${name}" "ace2-era5" "$dataset"
done
