#!/bin/bash
# 240-IC CM4 zero-shot coupled verification: 20 years (0231-0250) x 12 monthly
# ICs, 146 coupled steps (~24 months), one job per year. The ocean checkpoint
# under test is supplied per launch; the atmosphere is the fixed CM4 ACE.
#
#   OCEAN_CKPT_DS=<results dataset> NAME_PREFIX=samudra-enso-zs240-<arm> \
#     ./launch-zeroshot240-evals.sh
#
# CKPT_FILE selects which checkpoint inside the dataset (default: the
# inference-selected one).
set -euo pipefail
OCEAN_CKPT_DS="${OCEAN_CKPT_DS:?set to the training results dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01KJ70WK2NH4T2T4AVAAPYFSHA}"
ATMOS_CKPT_FILE="${ATMOS_CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME_PREFIX="${NAME_PREFIX:?set a job name prefix}"
YEARS="${YEARS:-0231 0232 0233 0234 0235 0236 0237 0238 0239 0240 0241 0242 0243 0244 0245 0246 0247 0248 0249 0250}"
PRIORITY="${PRIORITY:-urgent}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0
for Y in $YEARS; do
  JOB="${NAME_PREFIX}-yr${Y}"
  out=$(gantry run \
    --name "$JOB" --task-name "$JOB" \
    --description "240-IC zero-shot coupled verification, year ${Y}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-zs240 \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${OCEAN_CKPT_DS}:${CKPT_FILE}:/ocean_ckpt.tar" \
    --dataset "${ATMOS_CKPT_DS}:${ATMOS_CKPT_FILE}:/atmos_ckpt.tar" \
    --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
    --min-runtime "${MIN_RUNTIME:-8h}" \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator \
       "${SCRIPT_PATH}/wave1_eval_configs/zeroshot240/yr${Y}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) \
    || echo "FAILED $JOB: $(echo "$out" | tail -1)"
done
echo "${NAME_PREFIX}: launched $ok/$(echo $YEARS | wc -w)"
