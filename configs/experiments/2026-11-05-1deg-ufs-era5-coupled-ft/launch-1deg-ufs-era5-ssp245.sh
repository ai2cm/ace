#!/bin/bash
# SSP2-4.5 coupled run (fme.coupled.inference) of a 1deg UFS + ERA5 coupled fine-tune checkpoint; configs in ./ssp.
#   ARM=noohc-fto-mse CKPT_DS=<results ds> ./launch-1deg-ufs-era5-ssp245.sh
#   ARM=ohc-uhm1p7-fto-then-ftalt CKPT_DS=<results ds> ./launch-1deg-ufs-era5-ssp245.sh
# OCEAN_FORCING_DS: the 5-daily ocean static-forcing store (Beaker dataset holding ocean_forcing_ssp245.zarr).
set -euo pipefail
ARM="${ARM:?arm tag (ssp/ufs1deg-ssp245-<ARM>.yaml)}"
CKPT_DS="${CKPT_DS:?coupled fine-tune results dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_FORCING_DS="${OCEAN_FORCING_DS:-01M7AYRDF38RSF8X1Q08AZPNA4}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-ufs1deg-era5-${ARM}-ssp245-2015-2099"
C="ufs1deg-ssp245-${ARM}.yaml"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg UFS + ERA5 coupled SSP2-4.5 run 2015-2099 (${C}) of ${CKPT_DS}:${CKPT_FILE}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs1deg-era5-ssp245 \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --dataset "${OCEAN_FORCING_DS}:/oceanforcing/ocean_forcing_ssp245.zarr" \
  --gpus 1 --shared-memory 200GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.inference "${SCRIPT_PATH}/ssp/${C}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
