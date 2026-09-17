#!/bin/bash
# Coupled inference over 2022-2023: 24 monthly initializations the model never
# trained on (fine-tuning stops 2021-12-27; 2022 onward was the validation
# split, so no gradient updates). Inference rather than evaluation because the
# rollouts run past the end of the ocean dataset, so verification comes later
# from an observational SST product instead of from paired targets.
set -euo pipefail
CKPT_DS="${CKPT_DS:-01M2GMQC7S44NCJN3NCS3CFEVV}"
CKPT_FILE="${CKPT_FILE:-best_inference_ckpt.tar}"
BLOCKS="${BLOCKS:-1 2 3 4}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0
for B in $BLOCKS; do
JOB="samudra-ufs-cplft7-fc2022-blk${B}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "Coupled inference, 2022-2023 initializations, block ${B}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority high --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-enso-fc2022 \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.inference "${SCRIPT_PATH}/wave1_eval_configs/ufs-era5-cplft7-2022/blk${B}.yaml" 2>&1)
echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "2022-2023 inference launched: $ok/$(echo $BLOCKS | wc -w)"
