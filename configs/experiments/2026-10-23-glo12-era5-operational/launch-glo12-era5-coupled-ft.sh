#!/bin/bash
# GLO12 operational coupled fine-tune: UFS-trained SamudrACE ocean + ERA5 4deg ACE2S (both lifted from the alternating UFS coupled FT) on the GLO12 4deg 5-day store with ERA5, 2022-06..2024-12 (2025 held out).
#   ./launch-glo12-era5-coupled-ft.sh   (defaults = the lifted spb-ufsstats components, GLO12 derived store, GLO12 stats, GLO12-period ERA5 store)
set -euo pipefail
OCEAN_CKPT_DS="${OCEAN_CKPT_DS:-01M5MT2QD751XNNY94JB6JKMZV}"   # ocean lifted from the alternating UFS coupled FT
OCEAN_CKPT_FILE="${OCEAN_CKPT_FILE:-best_inference_ckpt.tar}"
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01M5MT6MCQ7Q0JTK2YKWJ5G0CM}"   # atmosphere lifted from the same coupled FT
ATMOS_CKPT_FILE="${ATMOS_CKPT_FILE:-best_inference_ckpt.tar}"
OCEAN_STATS_DS="${OCEAN_STATS_DS:-01M5K93QN20V8BPRWDEFF550RP}"   # GLO12 2022-06..2024-12 ocean stats (+ UFS placeholder entries)
OCEAN_STATS_SUBDIR="${OCEAN_STATS_SUBDIR:-}"
OCEAN_DS="${OCEAN_DS:-01M5K92VSR43C7KBSJBTCRPF9C}"   # derived GLO12 4deg store (zero placeholders sfdsi/UI/VI)
ERA5_DS="${ERA5_DS:-01M5K93RXH7XB0MJM1H41X2BE1}"   # era5-4deg-8layer-daily-glo12-coupled-2022-2025
ERA5_ZARR=2026-10-23-era5-4deg-8layer-daily-glo12-coupled-2022-2025.zarr
CFG="${CFG:-glo12-era5-coupledft-spb}"
N_GPUS="${N_GPUS:-2}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-${CFG}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "GLO12 4deg ocean + ERA5 4deg daily ACE2S (lifted from the UFS alternating coupled FT): coupled fine-tune ${CFG}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "${MIN_RUNTIME:-8h}" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-glo12-era5-coupled-ft \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${OCEAN_STATS_DS}:${OCEAN_STATS_SUBDIR:+${OCEAN_STATS_SUBDIR}:}/ocean_stats" \
  --dataset "${OCEAN_CKPT_DS}:${OCEAN_CKPT_FILE}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_CKPT_DS}:${ATMOS_CKPT_FILE}:/atmos_ckpt.tar" \
  --dataset "${OCEAN_DS}:/glo12/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025-cm4vars.zarr" \
  --dataset "${ERA5_DS}:/era5glo/${ERA5_ZARR}" \
  --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${SCRIPT_PATH}/${CFG}.yaml" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || echo "FAILED $JOB: $(echo "$out" | tail -3)"
