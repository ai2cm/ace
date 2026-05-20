#!/bin/bash

set -e

SCRIPT_DIR=$(dirname "$0")

EXISTING_RESULTS_DATASETS=(
    "01KS0YYM4RVW8Q9F26XW7K97SZ"
    "01KS0SGJNRYDYAVJMVZS27TNXA"
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
