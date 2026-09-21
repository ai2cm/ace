#!/bin/bash
# Coupled fine-tune of the bounded 4deg residual-anchor ocean arms against the
# frozen 4deg ACE2S atmosphere: the 1deg coupled-FT protocol at 4deg, so we can
# ask whether coupled fine-tuning removes the ocean's piControl heat drift (and
# compare against the no-FT coupled control from launch-4deg-coupled-evals.sh).
#   ARMS="ff-ohc resid-ohc resid-anomohc" ./launch-4deg-coupled-ft.sh
set -euo pipefail
ARMS="${ARMS:-ff-ohc resid-ohc resid-anomohc}"
COUPLED_STATS="${COUPLED_STATS:-01KY8D816E67NXHP2EB29FHBJM}"   # 4deg cm4 coupled stats: coupled_atmosphere/ + ocean/
ATMOS_DS="${ATMOS_DS:-01M0YFRM5TKQR6GSNDXKJSBNAS}"           # 4deg atmos_ace2s-ft rs0 results
ATMOS_CKPT="${ATMOS_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_CKPT="${OCEAN_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
N_GPUS="${N_GPUS:-2}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  total=$((total+1))
  OCEAN_DS=$(beaker experiment get "troya/samudra-anchor4deg-${A}" --format json | jq -r '.[0].jobs[-1].result.beaker')
  [ -n "$OCEAN_DS" ] && [ "$OCEAN_DS" != "null" ] || { echo "no results dataset for $A"; continue; }
  JOB="samudra-anchor4deg-${A}-coupled-ft"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg anchor arm ${A}: coupled fine-tune (ocean MSE, atmosphere frozen) with the 4deg ACE2S atmosphere" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled-ft \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${COUPLED_STATS}:coupled_atmosphere:/atmos_stats" \
    --dataset "${COUPLED_STATS}:ocean:/ocean_stats" \
    --dataset "${OCEAN_DS}:${OCEAN_CKPT}:/ocean_ckpt.tar" \
    --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
    --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${SCRIPT_PATH}/coupled/${A}-coupled-ft.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled FT launched: $ok/$total"
