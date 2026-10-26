#!/bin/bash
# 1deg coupled fine-tune of the random-CO2 ACE2S rs1 with James's ocean_ppad_fmean Samudra (configs in ./coupled),
# resources as Elynn's 1deg alternating runs (4 GPUs, 840 GiB shared memory). Troy 2026-10-22.
#   ARM=alt ./launch-1deg-coupled-ft.sh            # alternating baseline
#   ARM=alt-replay ./launch-1deg-coupled-ft.sh     # alternating + random-CO2 replay
set -euo pipefail
ARM="${ARM:?alt or alt-replay}"
COUPLED_STATS="${COUPLED_STATS:-01KXNT0RA6VX2YTZ8WJ936Q5RS}"      # 1deg cm4 coupled stats: coupled_atmosphere/ + ocean/
OCEAN_DS="${OCEAN_DS:-01M4Q2J28N6YJXMWR1EE8Z1Q1D}"                # ocean_ppad_fmean rs0 results (wandb mnh7jo06)
OCEAN_CKPT="${OCEAN_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
ATMOS_DS="${ATMOS_DS:-01M55CYWQKSFSB8GBCVM6WK59P}"                # random-CO2 ACE2S origLR-ft rs1 results (wandb tb6xf0ju)
ATMOS_CKPT="${ATMOS_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
N_GPUS="${N_GPUS:-4}"
PRIORITY="${PRIORITY:-high}"
MIN_RUNTIME="${MIN_RUNTIME:-6h}"   # guaranteed time before Beaker may preempt (gantry --min-runtime; 0 = any time)
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="1deg-randco2rs1-fmean-${ARM}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg coupled FT (${ARM}) of the random-CO2 ACE2S rs1 with the ocean_ppad_fmean Samudra" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=1deg-randco2-replay \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${COUPLED_STATS}:coupled_atmosphere:/atmos_stats" \
  --dataset "${COUPLED_STATS}:ocean:/ocean_stats" \
  --dataset "${OCEAN_DS}:${OCEAN_CKPT}:/ocean_ckpt.tar" \
  --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
  ${EXISTING_RESULTS_DS:+--dataset "${EXISTING_RESULTS_DS}:/existing-results"} \
  --gpus "$N_GPUS" --shared-memory 840GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${SCRIPT_PATH}/coupled/1deg-randco2rs1-fmean-${ARM}${EXISTING_RESULTS_DS:+-resume}.yaml" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
