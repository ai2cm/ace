#!/bin/bash
# Multi-step fine-tune of the CM4-line ACE2.5 atmosphere on ERA5 (fme.ace.train, config ssp/ace2p5-era5-multistep-ft.yaml).
#   EXTRA_DS=<results dataset of launch-build-ace2p5-era5-extra.sh> ./launch-1deg-era5-ace2p5-ft.sh
#   [CKPT_DS=01M6DWYW0RWBN0X33ABS98T0AH CKPT_FILE=training_checkpoints/best_inference_ckpt.tar N_GPUS=4 EXISTING_RESULTS_DS=<ds to resume>]
set -euo pipefail
EXTRA_DS="${EXTRA_DS:?results dataset holding ace2p5_era5_extra.zarr}"
CKPT_DS="${CKPT_DS:-01M6DWYW0RWBN0X33ABS98T0AH}"                          # Spencer's CM4-line ACE2.5 rs1 (the SSP2-4.5 atmosphere-only run's checkpoint)
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
ATMOS_STATS_DS="${ATMOS_STATS_DS:-01KXNT0RA6VX2YTZ8WJ936Q5RS}"             # CM4 stats, coupled_atmosphere/ = ACE2.5's own normalization
N_GPUS="${N_GPUS:-4}"
PRIORITY="${PRIORITY:-high}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
CFG_NAME="${CFG_NAME:-ace2p5-era5-multistep-ft}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="ace-1deg-era5-${CFG_NAME}${JOB_SUFFIX:-}"
CFG="${SCRIPT_PATH}/ssp/${CFG_NAME}.yaml"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "CM4-line ACE2.5 (${CKPT_DS}:${CKPT_FILE}) multi-step fine-tune on ERA5 1deg 6-hourly, summer-2026 ACE2S recipe (${CFG})" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=era5-1deg-ace2p5-ft \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${ATMOS_STATS_DS}:coupled_atmosphere:/atmos_stats" \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --dataset "${EXTRA_DS}:ace2p5_era5_extra.zarr:/extra/ace2p5_era5_extra.zarr" \
  ${EXISTING_RESULTS_DS:+--dataset "${EXISTING_RESULTS_DS}:/existing-results"} \
  --gpus "$N_GPUS" --shared-memory 400GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.ace.train "${CFG}" ${EXISTING_RESULTS_DS:+--override resume_results.existing_dir=/existing-results resume_results.resume_wandb=true} 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
