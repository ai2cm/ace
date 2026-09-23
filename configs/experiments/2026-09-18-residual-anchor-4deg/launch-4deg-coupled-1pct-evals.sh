#!/bin/bash
# 130-year 1pctCO2 COUPLED rollouts at 4deg (start 0001-01-06, 9488 coupled steps), the ramp
# counterpart of the 200-yr piControl coupled evals, so each arm gets Elynn's two drift figures
# (constant CO2 and ramp) in coupled mode. Two forms per arm:
#   noft : the ocean-only arm's best_inference_ckpt + the frozen 4deg ACE2S atmosphere (two-checkpoint config)
#   ft   : the coupled fine-tune's own best_inference_ckpt (single-checkpoint config)
# Usage: ARMS="ff-ohc resid-ohc resid-anomohc" FORMS="noft ft" ./launch-4deg-coupled-1pct-evals.sh
set -euo pipefail
ARMS="${ARMS:-ff-ohc resid-ohc resid-anomohc}"
FORMS="${FORMS:-noft ft}"
PRIORITY="${PRIORITY:-high}"
ATMOS_DS="${ATMOS_DS:-01M0YFRM5TKQR6GSNDXKJSBNAS}"   # cm4_1pct_46to125_piC_156to235_4deg-atmos_ace2s-ft
declare -A FTNAME=([ff-ohc]="samudra-anchor4deg-ff-ohc-coupled-ft-52d6" [resid-ohc]="samudra-anchor4deg-resid-ohc-coupled-ft-1fbe" [resid-anomohc]="samudra-anchor4deg-resid-anomohc-coupled-ft-cbad")
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  for F in $FORMS; do
    total=$((total+1))
    if [ "$F" = "noft" ]; then
      DS=$(beaker experiment get "troya/samudra-anchor4deg-${A}" --format json | jq -r '.[0].jobs[-1].result.beaker')
      MOUNTS=(--dataset "${DS}:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar" --dataset "${ATMOS_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar")
      CFG=coupled-evaluator-config-1pct-ic0001-130yr-4deg.yaml
    else
      DS=$(beaker experiment get "troya/${FTNAME[$A]}" --format json | jq -r '.[0].jobs[-1].result.beaker')
      MOUNTS=(--dataset "${DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar")
      CFG=coupled-evaluator-config-1pct-ic0001-130yr-4deg-singleckpt.yaml
    fi
    [ -n "$DS" ] && [ "$DS" != "null" ] || { echo "no results dataset for $A/$F"; continue; }
    JOB="samudra-anchor4deg-${A}-coupled-${F}-eval-1pct-ic0001-130yr"
    out=$(gantry run --name "$JOB" --task-name "$JOB" \
      --description "4deg anchor arm ${A} (${F}): 130-yr 1pctCO2 coupled rollout" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
      --weka climate-default:/climate-default \
      --env PYTORCH_ALLOC_CONF=expandable_segments:True \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
      --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      "${MOUNTS[@]}" \
      --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
    echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
  done
done
echo "coupled 1pct evals launched: $ok/$total"
