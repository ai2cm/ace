#!/bin/bash
# Run the blowup rollout from the best-validation checkpoint, writing output to
# output/run (autoregressive_predictions.nc etc.).
#
# Prerequisites (run once, from this directory):
#   ./download_checkpoint.sh        # -> checkpoint/best_inference_ckpt.tar
#   python prepare_data.py          # -> data/era5_4deg_blowup_slice.nc
#
# inference.yaml reads the staged local netCDF by default (no weka, no
# forkserver). To stream directly from GCS instead, edit inference.yaml as noted
# in its forcing_loader comment.
#
# Usage: ./run_inference.sh [n_forward_steps]   (default 2922, ~8 years)
set -e
cd "$(dirname "$0")"

N_STEPS="${1:-2922}"

python -m fme.ace.inference.inference inference.yaml \
    --override "n_forward_steps=${N_STEPS}"

echo "predictions at $(pwd)/output/run/autoregressive_predictions.nc"
