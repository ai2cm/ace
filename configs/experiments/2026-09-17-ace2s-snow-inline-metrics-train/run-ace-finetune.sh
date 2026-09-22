#!/bin/bash
# Stage 2: multi-step fine-tuning from a finished stage-1 run. Each arm needs the
# Beaker result dataset of its 1-step job, which is mounted at /weights; the config
# reads training_checkpoints/best_ckpt.tar from there.
#
#   ./run-ace-finetune.sh ARM CHECKPOINT_DATASET [ARM CHECKPOINT_DATASET ...]
#   ./run-ace-finetune.sh cm4-control 01M2VVJ5A75WKXQXJVS4XEMT4Y era5-control 01M2TZCJAT224Z4KBKJFJB8TGQ
#
# The checkpoint datasets used for each launch are recorded in the README.

source "$(dirname "$0")/launch-common.sh"

if [[ $# -eq 0 || $(( $# % 2 )) -ne 0 ]]; then
  echo "usage: $0 ARM CHECKPOINT_DATASET [ARM CHECKPOINT_DATASET ...]" >&2; exit 1
fi

while [[ $# -gt 0 ]]; do
  arm="$1"; ckpt="$2"; shift 2
  dataset="${arm%%-*}"        # cm4 or era5
  treatment="${arm#*-}"       # control or masked-naive
  config="$arm-multi-step-finetune-daily.yaml"
  if [[ ! -f "$SCRIPT_PATH/$config" ]]; then
    echo "no fine-tuning config $config for arm $arm" >&2; exit 1
  fi
  run_training "$config" \
    "ace2s-snowmetrics-$dataset-daily-$treatment-multi-step-finetune-rs0" \
    --ckpt "$ckpt" "seed=0"
done
