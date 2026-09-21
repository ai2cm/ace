#!/bin/bash
# Coupled 200-year piControl rollouts of the 4deg residual-anchor arms with the
# frozen 4deg ACE2S atmosphere and NO coupled fine-tuning, via the coupled
# evaluator's two-checkpoint form. The control for "does coupling remove the
# ocean's heat drift": same ocean weights as the ocean-only centennial rollouts,
# now with an interactive atmosphere.
#   ARMS="ff-ohc resid-ohc resid-anomohc" ./launch-4deg-coupled-evals.sh
set -euo pipefail
ARMS="${ARMS:-ff-ohc resid-ohc resid-anomohc}"
# cm4_1pct_46to125_piC_156to235_4deg-atmos_ace2s-ft rs0 (beaker 01M0V1PX7ZYC0AJTZ24W8MAH5F)
ATMOS_DS="${ATMOS_DS:-01M0YFRM5TKQR6GSNDXKJSBNAS}"
ATMOS_CKPT="${ATMOS_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_CKPT="${OCEAN_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
CFG="${CFG:-coupled-evaluator-config-piC-ic0151-200yr-4deg.yaml}"
TAG="${TAG:-piC-ic0151-200yr}"
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
  JOB="samudra-anchor4deg-${A}-coupled-noft-eval-${TAG}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg anchor arm ${A} coupled to the frozen 4deg ACE2S atmosphere, no coupled FT: ${TAG}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${OCEAN_DS}:${OCEAN_CKPT}:/ocean_ckpt.tar" \
    --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled no-FT evals launched: $ok/$total"
