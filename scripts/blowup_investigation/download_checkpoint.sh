#!/bin/bash
# Fetch the best-validation (EMA) inference checkpoint from the training run
# 01KTCTN1TCJ89NGQCMTJ9GWC2K (train-4deg-daily-v1-era5-only-residual-rs0) into
# ./checkpoint. Normalization stats are baked into the checkpoint, so no
# /climate-default mount is needed to load the stepper.
set -e
cd "$(dirname "$0")"

RESULT_DATASET=01KTCTN1ZYTC2QHBJCEAGEYT86  # result dataset of the training run
mkdir -p checkpoint
beaker dataset fetch "$RESULT_DATASET" \
    --prefix training_checkpoints/best_inference_ckpt.tar \
    -o /tmp/blowup_ckpt_dl
mv /tmp/blowup_ckpt_dl/training_checkpoints/best_inference_ckpt.tar checkpoint/
echo "checkpoint at $(pwd)/checkpoint/best_inference_ckpt.tar"
