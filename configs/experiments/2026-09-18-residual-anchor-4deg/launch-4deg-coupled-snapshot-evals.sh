#!/bin/bash
# Coupled centennial rollouts (200-yr piControl, 130-yr 1pctCO2) from a single coupled checkpoint
# held in a snapshot dataset, e.g. a mid-training snapshot of a from-scratch coupled arm.
#   ARM=ff-ohc-coupled-scratch CKPT_DS=<dataset> [CKPT_FILE=training_checkpoints/best_inference_ckpt.tar]
#   [JOB_SUFFIX=-snap0928] SCENARIOS="piC 1pct" ./launch-4deg-coupled-snapshot-evals.sh
set -euo pipefail
ARM="${ARM:?arm name for the job}"
CKPT_DS="${CKPT_DS:?snapshot dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
JOB_SUFFIX="${JOB_SUFFIX:-}"
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
    piC)  CFG=${CFG_PIC:-coupled-evaluator-config-piC-ic0151-200yr-4deg-singleckpt.yaml};  TAG=piC-ic0151-200yr ;;
    1pct) CFG=coupled-evaluator-config-1pct-ic0001-130yr-4deg-singleckpt.yaml; TAG=1pct-ic0001-130yr ;;
    *) echo "unknown scenario $S"; exit 1 ;;
  esac
  JOB="samudra-anchor4deg-${ARM}${JOB_SUFFIX}-coupled-eval-${TAG}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg ${ARM}${JOB_SUFFIX}: ${TAG} coupled rollout from a snapshot coupled checkpoint" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled snapshot evals launched: $ok/$total"
