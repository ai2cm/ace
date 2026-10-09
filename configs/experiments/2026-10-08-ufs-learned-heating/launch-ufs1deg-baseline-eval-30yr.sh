#!/bin/bash
# Full-record rollout of the July 2026 1deg UFS baseline checkpoint (weka store), for comparison with the 4deg arms.
set -euo pipefail
CKPT_DS="${CKPT_DS:-01KX6MCJ53X0VCK5T880P6TEG1}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME="${NAME:-ufs1deg-baseline-jul2026}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-${NAME}-eval-30yr-1994"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 1deg July-2026 baseline: full-record ocean-only rollout from 1994 with monthly outputs" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs4deg-eval-10yr \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --weka climate-default:/climate-default \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval/evaluator-config-ufs1deg-oldstore-30yr-1994.yaml" 2>&1)
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || echo "FAILED $JOB: $(echo "$out" | tail -3)"
