#!/bin/bash
# AMIP-style 1979-2025 runs (ERA5 SST) of a CM4-line ACE2.5 atmosphere with historical CO2 and with CO2 held at its 1979 value (configs ssp/ace2p5-amip-1979-2025-co2-<member>.yaml).
#   TAG=ace2p5cm4era5 CKPT_DS=01M7N75EC6EW8V3P6QAP0YB9GS CKPT_FILE=training_checkpoints/best_inference_ckpt.tar ./launch-1deg-era5-ace2p5-amip-co2.sh
set -euo pipefail
TAG="${TAG:?checkpoint tag for the job name}"
CKPT_DS="${CKPT_DS:?atmosphere checkpoint dataset}"
CKPT_FILE="${CKPT_FILE:?checkpoint file inside the dataset}"
LEVELS="${LEVELS:-actual 1979}"
EXTRA_DS="${EXTRA_DS:-01M7BHZJYYNHVJ37KAZE4QEZBS}"   # ace2p5_era5_extra.zarr (carbon_dioxide ppmv + lake_fraction on the ERA5 calendar)
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for PPM in $LEVELS; do
  JOB="ace-1deg-era5-${TAG}-amip-1979-2025-co2-${PPM}"
  C="ace2p5-amip-1979-2025-co2-${PPM}.yaml"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg atmosphere-only AMIP 1979-2025 run, CO2 member ${PPM} (${C}) of ${CKPT_DS}:${CKPT_FILE}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
    --weka climate-default:/climate-default \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=era5-1deg-ace2p5-amip-co2 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --dataset "${EXTRA_DS}:ace2p5_era5_extra.zarr:/extra/ace2p5_era5_extra.zarr" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "${SCRIPT_PATH}/ssp/${C}" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
done
