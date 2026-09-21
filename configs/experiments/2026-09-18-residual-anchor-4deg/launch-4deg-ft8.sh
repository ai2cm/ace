#!/bin/bash
# 8-step fine-tune stage for the bounded 4deg residual-anchor arms, matching the
# 1deg centennial protocol's two-stage recipe (75 ep @ 4 steps, then 10 ep @ 8
# steps from best_ckpt) so the centennial heat numbers compare like-for-like.
# Ocean-only, 1 GPU per arm.
#   ARMS="ff-ohc resid-ohc resid-anomohc" ./launch-4deg-ft8.sh
set -euo pipefail
STATS_DS="${STATS_DS:-01KY8D816E67NXHP2EB29FHBJM}"   # 4deg cm4 coupled stats
ARMS="${ARMS:-ff-ohc resid-ohc resid-anomohc}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  total=$((total+1))
  CKPT_DS=$(beaker experiment get "troya/samudra-anchor4deg-${A}" --format json | jq -r '.[0].jobs[-1].result.beaker')
  [ -n "$CKPT_DS" ] && [ "$CKPT_DS" != "null" ] || { echo "no results dataset for $A"; continue; }
  JOB="samudra-anchor4deg-${A}-ft8"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg anchor arm ${A}: 10-epoch 8-step fine-tune from its best_ckpt" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-anchor-4deg-ft8 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${STATS_DS}:ocean:/ocean_stats" \
    --dataset "${CKPT_DS}:training_checkpoints/best_ckpt.tar:/ckpt.tar" \
    --gpus 1 --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node=1 -m fme.ace.train "${SCRIPT_PATH}/ft8/${A}-ft8.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "4deg ft8 launched: $ok/$total"
