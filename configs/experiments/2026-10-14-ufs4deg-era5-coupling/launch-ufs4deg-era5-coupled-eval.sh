#!/bin/bash
# Coupled UFS 4deg ocean + ERA5 4deg daily ACE2S atmosphere, two-checkpoint evaluator (no coupled fine-tune).
#   OCEAN_CKPT_DS=<results ds> ERA5_DS=<era5 coupled store ds> CFG=coupled-evaluator-config-ufs4deg-era5-smoke.yaml NAME=smoke ./launch-ufs4deg-era5-coupled-eval.sh
set -euo pipefail
OCEAN_CKPT_DS="${OCEAN_CKPT_DS:?ocean results dataset}"
OCEAN_CKPT_FILE="${OCEAN_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01M39XBD8K0KB3GAQ79ETVN470}"   # 4deg-daily-ace2s-ft3-detached-rs0 (Jeremy, wandb 1r7oyu58)
ATMOS_CKPT_FILE="${ATMOS_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_DS="${OCEAN_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"   # ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh (zarr at the dataset root)
ERA5_DS="${ERA5_DS:-01M4W630AM1HQRYKJ2AJ0JMNN1}"   # era5-4deg-8layer-daily-ufs-coupled-1994-2023-v2 (zarr contents at the dataset root)
ERA5_ZARR=2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr
CFG="${CFG:-coupled-evaluator-config-ufs4deg-era5-smoke.yaml}"
NAME="${NAME:-smoke}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-ufs4deg-era5-coupled-${NAME}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 4deg ocean + ERA5 4deg daily ACE2S, coupled evaluator: ${NAME}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs4deg-era5-coupled \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${OCEAN_CKPT_DS}:${OCEAN_CKPT_FILE}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_CKPT_DS}:${ATMOS_CKPT_FILE}:/atmos_ckpt.tar" \
  --dataset "${OCEAN_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr" \
  --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}" \   # the zarr contents sit at the dataset root
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/${CFG}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || echo "FAILED $JOB: $(echo "$out" | tail -3)"
