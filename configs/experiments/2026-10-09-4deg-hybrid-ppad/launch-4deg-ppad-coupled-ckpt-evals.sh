#!/bin/bash
# Centennial coupled rollouts from a coupled fine-tune's selected checkpoint (best_inference_ckpt), to check that the
# checkpoint the inline selection kept is stable over 200 yr piControl and 130 yr 1pctCO2 (Troy 2026-10-13: "as long
# as the ckpt that has the best inline inference is stable").
#   ARM=resid-ohc-cap0005-ppad-fmean VARIANT=-ens CKPT_DS=<results dataset holding training_checkpoints/best_inference_ckpt.tar> \
#     SCENARIOS="piC 1pct" ./launch-4deg-ppad-coupled-ckpt-evals.sh
set -euo pipefail
ARM="${ARM:?arm name, e.g. resid-ohc-cap0005-ppad-fmean}"
VARIANT="${VARIANT:--ens}"
CKPT_DS="${CKPT_DS:?results dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
SCENARIOS="${SCENARIOS:-piC 1pct}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for S in $SCENARIOS; do
  total=$((total+1))
  case "$S" in
    piC)  CFG=coupled-evaluator-config-piC-ic0151-200yr-4deg-singleckpt.yaml;  TAG=piC-ic0151-200yr ;;
    1pct) CFG=coupled-evaluator-config-1pct-ic0001-130yr-4deg-singleckpt.yaml; TAG=1pct-ic0001-130yr ;;
    *) echo "unknown scenario $S"; exit 1 ;;
  esac
  JOB="samudra-ppad4deg-${ARM}-coupled-ft${VARIANT}-eval-${TAG}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg ppad ${ARM} coupled-ft${VARIANT} selected checkpoint: ${TAG} coupled free rollout" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ppad-4deg-centennial \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB"; } || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled checkpoint evals launched: $ok/$total"
