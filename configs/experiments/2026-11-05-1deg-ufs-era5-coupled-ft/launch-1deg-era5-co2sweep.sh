#!/bin/bash
# Fixed-SST stepped-CO2 sweep of one 1deg ERA5 atmosphere checkpoint (fme.ace.inference; configs ssp/ace-co2sweep-<ppm>ppm.yaml).
#   TAG=ace2s6h  CKPT_DS=01M49ERVM1DVWV12G5DRDXM0PC CKPT_FILE=training_checkpoints/ema_ckpt_0002.tar ./launch-1deg-era5-co2sweep.sh
#   TAG=ace2era5 CKPT_DS=01J4MT10JPQ8MFA41F2AXGFYJ9 CKPT_FILE=training_checkpoints/best_inference_ckpt.tar ./launch-1deg-era5-co2sweep.sh
set -euo pipefail
TAG="${TAG:?checkpoint tag for the job name}"
CKPT_DS="${CKPT_DS:?atmosphere checkpoint dataset}"
CKPT_FILE="${CKPT_FILE:?checkpoint file inside the dataset}"
LEVELS="${LEVELS:-400 450 475 500 550 600}"
MIN_RUNTIME="${MIN_RUNTIME:-1h}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for PPM in $LEVELS; do
  JOB="ace-1deg-era5-${TAG}-co2sweep-${PPM}ppm-fixedsst-10yr"
  C="ace-co2sweep-${PPM}ppm.yaml"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg ERA5 atmosphere-only fixed-SST 10-yr run at ${PPM} ppm CO2 (${C}) of ${CKPT_DS}:${CKPT_FILE}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
    --weka climate-default:/climate-default \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=era5-1deg-co2sweep-fixedsst \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "${SCRIPT_PATH}/ssp/${C}" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
done
