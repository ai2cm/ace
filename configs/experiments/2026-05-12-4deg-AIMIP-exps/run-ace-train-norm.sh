#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/run-ace-train.sh"

JOB_GROUP="${1:-ace2-era5-norm}"

"$TRAIN_SCRIPT" "norm-base.yaml"         ace2-era5-train-4deg-norm-base         "$JOB_GROUP"
"$TRAIN_SCRIPT" "norm-gmst.yaml"         ace2-era5-train-4deg-norm-gmst         "$JOB_GROUP"
"$TRAIN_SCRIPT" "norm-respred.yaml"      ace2-era5-train-4deg-norm-respred      "$JOB_GROUP"
"$TRAIN_SCRIPT" "norm-gmst-respred.yaml" ace2-era5-train-4deg-norm-gmst-respred "$JOB_GROUP"
