#!/bin/bash
# 1deg coupled 200-yr piControl and 130-yr 1pctCO2 rollouts from a single coupled checkpoint, on the
# team's shared configs (configs/experiments/cm4_eval/coupled/ @fd29e2cac, copied under eval/).
#   CKPT_DS=<dataset> CKPT_PATH=training_checkpoints/best_inference_ckpt.tar JOB_PREFIX=<name> ./launch-coupled-evals-1deg.sh
set -euo pipefail
: "${CKPT_DS:?set CKPT_DS}"; : "${JOB_PREFIX:?set JOB_PREFIX}"
CKPT_PATH="${CKPT_PATH:-training_checkpoints/best_inference_ckpt.tar}"
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
    piC)  CFG=coupled-evaluator-config-piC-ic0151-200yr.yaml;  TAG=piC-ic0151-200yr ;;
    1pct) CFG=coupled-evaluator-config-1pct-ic0001-130yr.yaml; TAG=1pct-ic0001-130yr ;;
  esac
  JOB="${JOB_PREFIX}-coupled-eval-${TAG}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg coupled rollout ${TAG} of ${JOB_PREFIX} (team protocol)" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-coupled-ft-1deg \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_PATH}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled evals launched: $ok/$total"
