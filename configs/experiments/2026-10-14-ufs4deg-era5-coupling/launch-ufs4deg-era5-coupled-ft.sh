#!/bin/bash
# Coupled fine-tune of a UFS 4deg ocean with the frozen ERA5 4deg daily ACE2S atmosphere (configs from make_coupled_ft_config.py).
#   OCEAN_CKPT_DS=<ocean results ds> CFG=learnedheat-ftcm4-era5-coupled-ft-ens ./launch-ufs4deg-era5-coupled-ft.sh
set -euo pipefail
OCEAN_CKPT_DS="${OCEAN_CKPT_DS:?ocean results dataset}"
OCEAN_CKPT_FILE="${OCEAN_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01M39XBD8K0KB3GAQ79ETVN470}"   # 4deg-daily-ace2s-ft3-detached-rs0
ATMOS_CKPT_FILE="${ATMOS_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_STATS_DS="${OCEAN_STATS_DS:-01M5D5GG2FB3MWWDD5GGSH8QR2}"   # UFS 4deg stats + unaccounted_heating (+ CM4 placeholders); coupled stages use UFS ocean + ERA5 atmosphere stats (Troy 2026-10-20)
OCEAN_STATS_SUBDIR="${OCEAN_STATS_SUBDIR:-}"   # "ocean" for the CM4 stats dataset 01M4W3MH28S5HPAJ84R7AB4V3W (the 2026-10-14 runs; CM4 stats are for pretraining only)
OCEAN_DS="${OCEAN_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"   # ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh
ERA5_DS="${ERA5_DS:-01M4W630AM1HQRYKJ2AJ0JMNN1}"     # era5-4deg-8layer-daily-ufs-coupled-1994-2023-v2
ERA5_ZARR=2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr
CFG="${CFG:-learnedheat-ftcm4-era5-coupled-ft-ens}"
N_GPUS="${N_GPUS:-2}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-ufs4deg-era5-${CFG}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 4deg ocean + frozen ERA5 4deg daily ACE2S: coupled fine-tune ${CFG}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs4deg-era5-coupled-ft \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${OCEAN_STATS_DS}:${OCEAN_STATS_SUBDIR:+${OCEAN_STATS_SUBDIR}:}/ocean_stats" \
  --dataset "${OCEAN_CKPT_DS}:${OCEAN_CKPT_FILE}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_CKPT_DS}:${ATMOS_CKPT_FILE}:/atmos_ckpt.tar" \
  --dataset "${OCEAN_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr" \
  --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}" \
  --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${SCRIPT_PATH}/${CFG}.yaml" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || echo "FAILED $JOB: $(echo "$out" | tail -3)"
