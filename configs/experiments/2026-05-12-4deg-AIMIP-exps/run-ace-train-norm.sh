#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

JOB_GROUP="${1:-ace2-era5-norm}"

# run-ace-train.sh builds CONFIG_PATH as $SCRIPT_PATH$config_filename, where
# $SCRIPT_PATH comes from git rev-parse --show-prefix (CWD-relative). Since we
# cd to repo root above, SCRIPT_PATH is empty, so we pass absolute paths here.
"$SCRIPT_DIR/run-ace-train.sh" "$SCRIPT_DIR/norm-base.yaml"         ace2-era5-train-4deg-norm-base         "$JOB_GROUP"
"$SCRIPT_DIR/run-ace-train.sh" "$SCRIPT_DIR/norm-gmst.yaml"         ace2-era5-train-4deg-norm-gmst         "$JOB_GROUP"
"$SCRIPT_DIR/run-ace-train.sh" "$SCRIPT_DIR/norm-respred.yaml"      ace2-era5-train-4deg-norm-respred      "$JOB_GROUP"
"$SCRIPT_DIR/run-ace-train.sh" "$SCRIPT_DIR/norm-gmst-respred.yaml" ace2-era5-train-4deg-norm-gmst-respred "$JOB_GROUP"
