#!/bin/bash

set -e

CONFIG_FILENAME="${1:-ace-eval-config-4deg-AIMIP.yaml}"
JOB_GROUP="${2:-ace2-era5}"
SCRIPT_DIR=$(dirname "$0")
WANDB_PROJECT=${WANDB_PROJECT:-GMST_ResPred}

# List of (EXISTING_RESULTS_DATASET, JOB_NAME) pairs to evaluate
JOBS=(
  "01KSJG2N2J06AWXRFN94QNSEB8 ace2-era5-eval-4deg-norm-gmst-respred"
  "01KSNHVEM76HJY67Y5DBGDTCDH ace2-era5-eval-4deg-norm-respred"
  "01KSJG2N2J06AWXRFN94QNSEB8 ace2-era5-eval-4deg-norm-gmst"
  "01KSJG2DVBQ30AA2ZQ2VSB20H0 ace2-era5-eval-4deg-norm-base"
)

for entry in "${JOBS[@]}"; do
  EXISTING_RESULTS_DATASET=$(echo "$entry" | awk '{print $1}')
  JOB_NAME=$(echo "$entry" | awk '{print $2}')

  echo "Submitting eval job: $JOB_NAME (dataset: $EXISTING_RESULTS_DATASET)"
  "$SCRIPT_DIR/run-ace-eval.sh" "$CONFIG_FILENAME" "$JOB_NAME" "$JOB_GROUP" "$EXISTING_RESULTS_DATASET"
done
