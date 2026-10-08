#!/bin/bash
# UFS 4deg from-scratch oceans that LEARN the unaccounted column heating (see plan.md).
#   DATA_DS=<…-cm4vars-uh> STATS_DS=<…-stats-2026-10-06-uh> ./launch-ufs4deg-learnedheat.sh
set -euo pipefail
ARMS="${ARMS:-resid-cap0005-learnedheat ff-ohc-learnedheat}"
DATA_DS="${DATA_DS:?training dataset with unaccounted_heating fields (…-cm4vars-uh)}"
STATS_DS="${STATS_DS:?stats dataset with the unaccounted_heating entries (…-stats-2026-10-06-uh)}"
PRIORITY="${PRIORITY:-high}"
N_GPUS="${N_GPUS:-1}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  total=$((total+1))
  JOB="samudra-ufs4deg-${A}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "UFS 4deg replay: ${A} trained from scratch on UFS only" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs4deg-learnedheat \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${STATS_DS}:/ocean_stats" \
    --dataset "${DATA_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr" \
    --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "${SCRIPT_PATH}/${A}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "UFS 4deg from-scratch launched: $ok/$total"
