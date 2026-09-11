#!/bin/bash
# Ocean-only ENSO hindcasts on the UFS replay holdout (2002-2011): 99 monthly
# ICs x 128 five-day steps (~21 months), prescribed observed atmospheric
# forcing. Scores the UFS fine-tune's ocean component; NOT a coupled forecast
# (the prescribed wind stress carries ENSO information).
# CKPT_DS / CKPT_FILE select the checkpoint (default: 199b epoch-40 best_inference).
set -euo pipefail
CKPT_DS="${CKPT_DS:-01M28WK7RSGR0DRJCNFCF4DQZ7}"
CKPT_FILE="${CKPT_FILE:-best_inference_ckpt.tar}"
YEARS="${YEARS:-2002 2003 2004 2005 2006 2007 2008 2009 2010}"
PRIORITY="${PRIORITY:-urgent}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for Y in $YEARS; do
JOB="samudra-enso-ufsft40-holdout-yr${Y}"
gantry run \
  --name "$JOB" --task-name "$JOB" \
  --description "UFS FT epoch-40 holdout hindcasts, year ${Y} ICs, ocean-only, prescribed forcing" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-ufs-evals \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "${CKPT_DS}:training_checkpoints/${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/wave1_eval_configs/hybridufsft40/yr${Y}.yaml"
done
