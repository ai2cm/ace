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

# Job names mirror stage 1: the masked-naive arms carry "-land-snow" because their
# 1-step runs were trained on the per-land-area snow channels.
job_name_for() {
  local dataset="${1%%-*}" treatment="${1#*-}"
  case "$treatment" in
    control)      echo "ace2s-snowmetrics-$dataset-daily-control-multi-step-finetune-rs0" ;;
    masked-naive) echo "ace2s-snowmetrics-$dataset-daily-masked-naive-land-snow-multi-step-finetune-rs0" ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  arm="$1"; ckpt="$2"; shift 2
  config="$arm-multi-step-finetune-daily.yaml"
  if [[ ! -f "$SCRIPT_PATH/$config" ]]; then
    echo "no fine-tuning config $config for arm $arm" >&2; exit 1
  fi
  job_name_for "$arm" >/dev/null || { echo "unknown arm $arm" >&2; exit 1; }
  run_training "$config" "$(job_name_for "$arm")" --ckpt "$ckpt" "seed=0"
done
