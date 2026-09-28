#!/bin/bash
# 240-IC coupled forecast verification at 4deg: piControl years 0241-0260 x 12 monthly ICs,
# 24-month forecasts, one job per year. Two checkpoint forms:
#   FORM=two    OCEAN_CKPT_DS=<dataset> [OCEAN_CKPT_FILE=...]  (ocean-only ckpt + frozen 4deg ACE2S atmosphere)
#   FORM=single CKPT_DS=<dataset> [CKPT_FILE=...]              (one coupled checkpoint)
#   NAME_PREFIX=samudra-zs240-4deg-<arm> ./launch-4deg-zs240.sh
set -euo pipefail
FORM="${FORM:?two|single}"
NAME_PREFIX="${NAME_PREFIX:?job name prefix}"
ATMOS_DS="${ATMOS_DS:-01M0YFRM5TKQR6GSNDXKJSBNAS}"
ATMOS_CKPT_FILE="${ATMOS_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
YEARS="${YEARS:-$(seq -f '%04g' 241 260 | tr '\n' ' ')}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
if [ "$FORM" = "two" ]; then
  MOUNTS=(--dataset "${OCEAN_CKPT_DS:?}:${OCEAN_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}:/ocean_ckpt.tar"
          --dataset "${ATMOS_DS}:${ATMOS_CKPT_FILE}:/atmos_ckpt.tar")
else
  MOUNTS=(--dataset "${CKPT_DS:?}:${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}:/ckpt.tar")
fi
ok=0
for Y in $YEARS; do
  JOB="${NAME_PREFIX}-yr${Y}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg 240-IC coupled forecast verification, piControl year ${Y} (${NAME_PREFIX})" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-zs240-4deg \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    "${MOUNTS[@]}" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/zs240/yr${Y}-${FORM}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -1)"
done
echo "${NAME_PREFIX}: launched $ok/$(echo $YEARS | wc -w)"
