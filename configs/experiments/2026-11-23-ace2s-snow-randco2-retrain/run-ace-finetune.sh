#!/bin/bash
# Stage 2: multi-step fine-tuning from a finished stage-1 run. Each run needs the
# Beaker result dataset of its 1-step job; its training_checkpoints/best_ckpt.tar
# is mounted where the config reads it.
#
#   ./run-ace-finetune.sh RUN CHECKPOINT_DATASET [RUN CHECKPOINT_DATASET ...]
#
# RUN is <arm>-rs<seed>, e.g. cm4-masked-naive-pic-1pct-randco2-rs1 or
# cm4-control-pic-1pct-rs1. The checkpoint datasets used for each launch are
# recorded in the README.

source "$(dirname "$0")/launch-common.sh"

if [[ $# -eq 0 || $(( $# % 2 )) -ne 0 ]]; then
  echo "usage: $0 RUN CHECKPOINT_DATASET [RUN CHECKPOINT_DATASET ...]" >&2; exit 1
fi

while [[ $# -gt 0 ]]; do
  run="$1"; ckpt="$2"; shift 2
  arm="${run%-rs*}"
  seed="${run##*-rs}"
  config="$arm-multi-step-finetune-daily.yaml"
  if [[ ! -f "$SCRIPT_PATH/$config" || ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "no fine-tuning config $config or no -rs<seed> suffix for run $run" >&2; exit 1
  fi
  run_training "$config" "ace2s-snowfix-${arm}-daily-multi-step-finetune-rs${seed}" --ckpt "$ckpt" "seed=$seed"
done
