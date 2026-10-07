#!/bin/bash
# 50-yr 1pctCO2 OCEAN-ONLY rollouts (CM4 atmosphere prescribed) of one ocean checkpoint with several seeds for
# the stochastic ocean: the uncoupled twin of the ocean-forcing ablation, used to ask whether the ice
# collapses happen without ACE and whether their timing is stochastic.
#   CKPT_DS=<dataset> NAME=ffstoch-pretrain SEEDS="0 1 2" ./launch-4deg-oceanonly-seeds.sh
set -euo pipefail
CKPT_DS="${CKPT_DS:?checkpoint dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME="${NAME:?short arm name for the job}"
SEEDS="${SEEDS:-0 1 2}"
PRIORITY="${PRIORITY:-high}"
CFG=evaluator-config-1pct-ic0001-50yr-4deg-oceanonly-seed.yaml
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for SEED in $SEEDS; do
  total=$((total+1))
  JOB="samudra-anchor4deg-${NAME}-oceanonly-1pct-ic0001-50yr-seed${SEED}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg ${NAME}: 50-yr 1pct ocean-only rollout, CM4 atmosphere prescribed, seed ${SEED}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-oceanonly-seeds \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval/${CFG}" --override "seed=${SEED}" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB"; } || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "ocean-only seed runs launched: $ok/$total"
