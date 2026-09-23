#!/bin/bash
# 20-yr 4deg piControl evals with the corrector step-diagnostics on, for each arm twice: correctors as
# trained ("on"), and with the OHC correction disabled at inference ("noohc"). Measures how much of
# the temperature field each recipe's corrector supplies, and what the network does unaided.
#   ARMS="ff-ohc ff-ohc-precorr resid-ohc resid-anomohc" MODES="on noohc" ./launch-4deg-corrdiag-evals.sh
set -euo pipefail
ARMS="${ARMS:-ff-ohc ff-ohc-precorr ff-ohc-regL1w1 resid-ohc resid-anomohc resid-anomohc-precorr}"
MODES="${MODES:-on noohc}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  DS=$(beaker experiment get "troya/samudra-anchor4deg-${A}" --format json | jq -r '.[0].jobs[-1].result.beaker')
  [ -n "$DS" ] && [ "$DS" != "null" ] || { echo "no results dataset for $A"; continue; }
  for M in $MODES; do
    total=$((total+1))
    OVR=(); [ "$M" = "noohc" ] && OVR=(--override "stepper_override.disable_corrections=[ocean_heat_content_correction]")
    JOB="samudra-anchor4deg-${A}-corrdiag-${M}-piC-20yr"
    out=$(gantry run --name "$JOB" --task-name "$JOB" \
      --description "4deg arm ${A}: 20-yr piC with corrector diagnostics, OHC correction ${M}" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
      --weka climate-default:/climate-default \
      --env PYTORCH_ALLOC_CONF=expandable_segments:True \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
      --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-corrdiag \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset "${DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
      --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval/evaluator-config-piC-ic0151-20yr-4deg-corrdiag.yaml" "${OVR[@]}" 2>&1)
    echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
  done
done
echo "corrdiag evals launched: $ok/$total"
