#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$0")

EXISTING_RESULTS_DATASETS=(
    "01KS8GG67QZF3B4FPSY5HHCV5J"
    "01KS8GGDBKH7B6AE8BCEK71JB8"
)

DATASET_NAMES=(
    "var-masking"
    "dataset-masking"
)

for i in "${!EXISTING_RESULTS_DATASETS[@]}"; do
    dataset="${EXISTING_RESULTS_DATASETS[$i]}"
    name="${DATASET_NAMES[$i]}"
    "$SCRIPT_DIR/run-ace-eval.sh" "ace-eval-config-4deg-AIMIP.yaml" "ace-eval-4deg-${name}" "ace2-era5" "$dataset"
done
