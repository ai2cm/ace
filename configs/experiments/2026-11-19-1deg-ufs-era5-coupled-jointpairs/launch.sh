#!/bin/bash
# 1deg UFS + ERA5 coupled fine-tune with the one-stage joint recipe (config from make_config.py). 8 GPUs, one sample per GPU.
#   ./launch.sh
#   VARIANT=randco2 ./launch.sh         # uncoupled atmosphere batches = CM4 random-CO2 runs
#   VARIANT=x6-randco2 ./launch.sh      # the same with the CM4 X6 learning-rate schedules (VARIANT=x6: ERA5 uncoupled batches)
#   EXISTING_RESULTS_DS=<results dataset> ./launch.sh   # continue a previous submission (the -resume config)
set -euo pipefail
ATMOS_DS="${ATMOS_DS:-01M7N75EC6EW8V3P6QAP0YB9GS}"                 # ACE2.5-CM4-ERA5-6HR, epoch 16 of wandb ace-samudra-coupled-cm4/9l60thd4
ATMOS_CKPT="${ATMOS_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_DS="${OCEAN_DS:-01M6HWDXRVCZC1W5HFD5WPM19Z}"                 # corrector-kept UFS ocean-only fine-tune (wandb 75s8184h)
OCEAN_CKPT="${OCEAN_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_STATS_DS="${OCEAN_STATS_DS:-01M6F9VRGMY0NYX19R37PRTW92}"     # 1deg UFS 2026-10-06 stats
ATMOS_STATS_DS="${ATMOS_STATS_DS:-01KXNT0RA6VX2YTZ8WJ936Q5RS:coupled_atmosphere}"   # CM4 stats: this atmosphere's own normalization
FRAC_DS="${FRAC_DS:-01M6SM21BWEG1DJ2XNM4WVARAJ}"                   # era5-1deg-6h-ufs-coupled-fractions-1994-2023
FRAC_ZARR=2026-11-05-era5-1deg-6h-ufs-coupled-fractions-1994-2023.zarr
N_GPUS="${N_GPUS:-8}"
PRIORITY="${PRIORITY:-high}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
VARIANT="${VARIANT:-}"   # randco2: uncoupled atmosphere batches from the CM4 random-CO2 runs
JOB="samudra-ufs1deg-era5-ohc-uhm1p7-ace2p5cm4-jointpairs${VARIANT:+-$VARIANT}${TAG:+-$TAG}"
CFG="${SCRIPT_PATH}/ufs1deg-era5-ace2p5cm4-jointpairs${VARIANT:+-$VARIANT}${EXISTING_RESULTS_DS:+-resume}.yaml"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg UFS + ERA5 coupled FT, one-stage joint recipe (ocean / joint pairs / uncoupled ${VARIANT:-ERA5} atmosphere), atmosphere ${ATMOS_DS}:${ATMOS_CKPT}, ocean ${OCEAN_DS}:${OCEAN_CKPT}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs1deg-era5-coupled-ft \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${OCEAN_STATS_DS}:/ocean_stats" \
  --dataset "${ATMOS_STATS_DS}:/atmos_stats" \
  --dataset "${OCEAN_DS}:${OCEAN_CKPT}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
  --dataset "${FRAC_DS}:/era5frac/${FRAC_ZARR}" \
  ${EXISTING_RESULTS_DS:+--dataset "${EXISTING_RESULTS_DS}:/existing-results"} \
  --gpus "$N_GPUS" --shared-memory 840GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${CFG}" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
