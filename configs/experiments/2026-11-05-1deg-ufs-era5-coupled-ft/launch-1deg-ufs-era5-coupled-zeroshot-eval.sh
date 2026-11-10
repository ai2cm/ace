#!/bin/bash
# Zero-shot full-record coupled rollout (two-checkpoint form) of a 1deg UFS ocean checkpoint + an ERA5 1deg 6-hourly ACE2S, 1 GPU.
#   OCEAN_DS=<ds> [OCEAN_CKPT=training_checkpoints/best_inference_ckpt.tar] ARM=ace2s6h|ace2s512 NAME=<label> FRAC_DS=<v2 fraction store> ./launch-1deg-ufs-era5-coupled-zeroshot-eval.sh
set -euo pipefail
ARM="${ARM:?ace2s6h or ace2s512}"
case "$ARM" in
  ace2s6h)  ATMOS_DS_DEFAULT=01M49ERVM1DVWV12G5DRDXM0PC; ATMOS_CKPT_DEFAULT=training_checkpoints/ema_ckpt_0002.tar ;;
  ace2s512) ATMOS_DS_DEFAULT=01KSVC6YS7C18SGYV4VPZYZ232; ATMOS_CKPT_DEFAULT=training_checkpoints/best_inference_ckpt.tar ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
ATMOS_DS="${ATMOS_DS:-$ATMOS_DS_DEFAULT}"; ATMOS_CKPT="${ATMOS_CKPT:-$ATMOS_CKPT_DEFAULT}"
OCEAN_DS="${OCEAN_DS:?ocean checkpoint dataset}"; OCEAN_CKPT="${OCEAN_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
NAME="${NAME:?label}"
FRAC_DS="${FRAC_DS:?surface-fraction store dataset}"
FRAC_ZARR=2026-11-05-era5-1deg-6h-ufs-coupled-fractions-1994-2023.zarr
CFG="${CFG:-coupled-evaluator-config-ufs1deg-era5-fullrecord-1994-twockpt.yaml}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-ufs1deg-era5-${NAME}-zeroshot-fullrecord-1994"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg UFS + ERA5 zero-shot coupled full-record rollout: ocean ${OCEAN_DS}:${OCEAN_CKPT}, atmosphere ${ARM} ${ATMOS_DS}:${ATMOS_CKPT}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs1deg-era5-coupled-eval \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${OCEAN_DS}:${OCEAN_CKPT}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
  --dataset "${FRAC_DS}:/era5frac/${FRAC_ZARR}" \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
