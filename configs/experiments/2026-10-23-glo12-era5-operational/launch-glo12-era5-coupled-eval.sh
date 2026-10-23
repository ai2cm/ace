#!/bin/bash
# GLO12 coupled evaluator (2025 hindcasts of the GLO12 coupled fine-tune): CKPT_DS=<results ds> CKPT_FILE=training_checkpoints/ckpt_NNNN.tar NAME=... ./launch-glo12-era5-coupled-eval.sh
#   OCEAN_CKPT_DS=<results ds> ERA5_DS=<era5 coupled store ds> CFG=coupled-evaluator-config-ufs4deg-era5-smoke.yaml NAME=smoke ./launch-ufs4deg-era5-coupled-eval.sh
set -euo pipefail
MIN_RUNTIME="${MIN_RUNTIME:-3h}"   # beaker minimum guaranteed runtime (team template: 1deg 6h, 4deg 3h); override per launch
OCEAN_CKPT_DS="${OCEAN_CKPT_DS:-}"   # two-checkpoint form; or set CKPT_DS for a coupled fine-tune checkpoint (single-checkpoint form)
CKPT_DS="${CKPT_DS:-}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
[ -n "$OCEAN_CKPT_DS" ] || [ -n "$CKPT_DS" ] || { echo "set OCEAN_CKPT_DS (two-checkpoint) or CKPT_DS (coupled checkpoint)"; exit 1; }
OCEAN_CKPT_FILE="${OCEAN_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01M39XBD8K0KB3GAQ79ETVN470}"   # 4deg-daily-ace2s-ft3-detached-rs0 (Jeremy, wandb 1r7oyu58)
ATMOS_CKPT_FILE="${ATMOS_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_DS="${OCEAN_DS:-01M5K92VSR43C7KBSJBTCRPF9C}"   # derived GLO12 4deg store
ERA5_DS="${ERA5_DS:-01M5K93RXH7XB0MJM1H41X2BE1}"   # era5-4deg-8layer-daily-glo12-coupled-2022-2025
ERA5_ZARR=2026-10-23-era5-4deg-8layer-daily-glo12-coupled-2022-2025.zarr
CFG="${CFG:-coupled-evaluator-config-glo12-era5-hindcast-2025-6mo.yaml}"
NAME="${NAME:-hindcast-2025-6mo}"
PRIORITY="${PRIORITY:-high}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"   # e.g. data_writer.ocean.names=[sst,...] for oceans without the learned-heating channel
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-glo12-era5-coupled-${NAME}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 4deg ocean + ERA5 4deg daily ACE2S, coupled evaluator: ${NAME}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs4deg-era5-coupled \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  ${OCEAN_CKPT_DS:+--dataset "${OCEAN_CKPT_DS}:${OCEAN_CKPT_FILE}:/ocean_ckpt.tar"} \
  ${OCEAN_CKPT_DS:+--dataset "${ATMOS_CKPT_DS}:${ATMOS_CKPT_FILE}:/atmos_ckpt.tar"} \
  ${CKPT_DS:+--dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar"} \
  --dataset "${OCEAN_DS}:/glo12/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025-cm4vars.zarr" \
  --dataset "${ERA5_DS}:/era5glo/${ERA5_ZARR}" \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/${CFG}" ${EXTRA_OVERRIDE:+--override "$EXTRA_OVERRIDE"} 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || echo "FAILED $JOB: $(echo "$out" | tail -3)"
