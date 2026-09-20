#!/bin/bash
# Centennial free rollouts for the 4deg residual-anchor arms, on the same two
# protocols Elynn's heat-drift report (reports PR #114) uses: 200-yr piControl
# from 0151-01-06 and 130-yr 1pctCO2 from 0001-01-06, from each arm's
# best_inference_ckpt. Ocean-only, atmosphere prescribed; 1 GPU per job.
#   ARMS="ff-ohc resid-ohc" SCENARIOS="piC 1pct" ./launch-4deg-centennial-evals.sh
set -euo pipefail
ARMS="${ARMS:-ff-ohc resid-ohc resid-noohc resid-uniformohc resid-uniform-shapeslow resid-uniform-shapefast resid-anomohc split-noohc split-ohc}"
SCENARIOS="${SCENARIOS:-piC 1pct}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  CKPT_DS=$(beaker experiment get "troya/samudra-anchor4deg-${A}" --format json | jq -r '.[0].jobs[-1].result.beaker')
  [ -n "$CKPT_DS" ] && [ "$CKPT_DS" != "null" ] || { echo "no results dataset for $A"; continue; }
  for S in $SCENARIOS; do
    total=$((total+1))
    case "$S" in
      piC)  CFG=evaluator-config-piC-ic0151-200yr-4deg.yaml;  TAG=piC-ic0151-200yr ;;
      1pct) CFG=evaluator-config-1pct-ic0001-130yr-4deg.yaml; TAG=1pct-ic0001-130yr ;;
      *) echo "unknown scenario $S"; exit 1 ;;
    esac
    JOB="samudra-anchor4deg-${A}-eval-${TAG}"
    out=$(gantry run --name "$JOB" --task-name "$JOB" \
      --description "4deg anchor arm ${A}: ${TAG} free rollout (PR #114 protocol)" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
      --weka climate-default:/climate-default \
      --env PYTORCH_ALLOC_CONF=expandable_segments:True \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
      --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-centennial \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset "${CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
      --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
    echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
  done
done
echo "centennial evals launched: $ok/$total"
