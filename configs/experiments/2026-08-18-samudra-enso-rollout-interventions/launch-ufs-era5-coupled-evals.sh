#!/bin/bash
# Zero-shot coupled UFS+ERA5 ENSO hindcasts: 99 ICs (2002-2010), 128 coupled
# steps (~21 months), one job per year. TRUE forecast skill: the atmosphere
# is the ERA5-trained ACE2S, not prescribed forcing.
set -euo pipefail
OCEAN_CKPT_DS="${OCEAN_CKPT_DS:-01M28WK7RSGR0DRJCNFCF4DQZ7}"
ATMOS_CKPT_DS="${ATMOS_CKPT_DS:-01KWD8DZVJFKYC5A9PNW8259GH}"
YEARS="${YEARS:-2002 2003 2004 2005 2006 2007 2008 2009 2010}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for Y in $YEARS; do
JOB="samudra-ufs-era5-coupled-yr${Y}"
gantry run \
  --name "$JOB" --task-name "$JOB" \
  --description "Zero-shot coupled UFS+ERA5 hindcasts, year ${Y} ICs" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority urgent --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-ufs-evals \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "${OCEAN_CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar" \
  --dataset "${ATMOS_CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar" \
  --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/wave1_eval_configs/ufs-era5-coupled/yr${Y}.yaml"
done
