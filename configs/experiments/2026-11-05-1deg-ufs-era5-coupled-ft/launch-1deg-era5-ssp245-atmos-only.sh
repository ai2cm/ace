#!/bin/bash
# Atmosphere-only SSP2-4.5 run (fme.ace.inference, prescribed SST from the SSP forcing store); configs in ./ssp.
#   ARM=ace2s6h-original CKPT_DS=01M49ERVM1DVWV12G5DRDXM0PC CKPT_FILE=training_checkpoints/ema_ckpt_0002.tar ./launch-1deg-era5-ssp245-atmos-only.sh
#   ARM=ohc-uhm1p7-fto-then-ftalt-atmosphere CKPT_DS=<lifted atmosphere ds> CKPT_FILE=atmos_ckpt.tar ./launch-1deg-era5-ssp245-atmos-only.sh
set -euo pipefail
ARM="${ARM:?arm tag (ssp/ace-ssp245-<ARM>.yaml)}"
CKPT_DS="${CKPT_DS:?atmosphere checkpoint dataset}"
CKPT_FILE="${CKPT_FILE:?checkpoint file inside the dataset}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="ace-1deg-era5-${ARM}-ssp245-atmos-only-2015-2099"
C="ace-ssp245-${ARM}.yaml"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg ACE2S atmosphere-only SSP2-4.5 run with prescribed SST 2015-2099 (${C}) of ${CKPT_DS}:${CKPT_FILE}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs1deg-era5-ssp245 \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 200GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.ace.inference "${SCRIPT_PATH}/ssp/${C}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
