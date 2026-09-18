#!/bin/bash
# Residual-anchor ablation at 4deg, ocean-only, 1 GPU per arm.
# Five arms differing by one thing each; see the header of any config.
set -euo pipefail
STATS_DS="${STATS_DS:-01KY8D816E67NXHP2EB29FHBJM}"   # elynn/2026-07-22-cm4-1pctco2-4deg-coupled-stats
ARMS="${ARMS:-ff-ohc resid-ohc resid-noohc split-ohc split-noohc}"
N_GPUS="${N_GPUS:-1}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0
for A in $ARMS; do
JOB="samudra-anchor4deg-${A}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "4deg residual-anchor ablation, arm ${A}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-anchor-4deg \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${STATS_DS}:ocean:/ocean_stats" \
  --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.ace.train "${SCRIPT_PATH}/${A}.yaml" 2>&1)
echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "4deg anchor ablation launched: $ok/$(echo $ARMS | wc -w)"
