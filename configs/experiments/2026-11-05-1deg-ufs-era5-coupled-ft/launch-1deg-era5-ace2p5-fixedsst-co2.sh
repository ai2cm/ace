#!/bin/bash
# Fixed-SST abrupt-CO2 runs (1x and 4x pre-industrial, and 400 ppm) of a CM4-line ACE2.5 atmosphere checkpoint (configs ssp/ace2p5-fixedsst-co2-<level>.yaml).
#   TAG=ace2p5cm4era5 CKPT_DS=01M7N75EC6EW8V3P6QAP0YB9GS CKPT_FILE=training_checkpoints/best_inference_ckpt.tar EXTRA_DS=01M7B2PMF6F1YC9DQCRTZ07A69 ./launch-1deg-era5-ace2p5-fixedsst-co2.sh
set -euo pipefail
TAG="${TAG:?checkpoint tag for the job name}"
CKPT_DS="${CKPT_DS:?atmosphere checkpoint dataset}"
CKPT_FILE="${CKPT_FILE:?checkpoint file inside the dataset}"
LEVELS="${LEVELS:-1xpi 400ppm 4xpi}"
EXTRA_DS="${EXTRA_DS:?extra forcing store dataset (carbon_dioxide ppmv + lake_fraction)}"
MIN_RUNTIME="${MIN_RUNTIME:-1h}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for PPM in $LEVELS; do
  JOB="ace-1deg-era5-${TAG}-fixedsst-co2-${PPM}-10yr"
  C="ace2p5-fixedsst-co2-${PPM}.yaml"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg ERA5 atmosphere-only fixed-SST 10-yr run at CO2 level ${PPM} (${C}) of ${CKPT_DS}:${CKPT_FILE}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
    --weka climate-default:/climate-default \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=era5-1deg-ace2p5-fixedsst-co2 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --dataset "${EXTRA_DS}:/ace2p5forcing/ace2p5_ssp245_forcing.zarr" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.inference "${SCRIPT_PATH}/ssp/${C}" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
done
