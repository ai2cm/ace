#!/bin/bash
# Training-period hindcasts for the drift climatology: 216 monthly
# initializations over 1994-2001 and 2012-2021, the years the 2002-2011
# verification window leaves free. One job per year, 12 initializations each,
# 128 coupled steps (~21 months). Same checkpoint and protocol as the
# verification hindcasts, so the climatology they define is directly
# applicable to them.
set -euo pipefail
CKPT_DS="${CKPT_DS:-01M2GMQC7S44NCJN3NCS3CFEVV}"
CKPT_FILE="${CKPT_FILE:-best_inference_ckpt.tar}"
YEARS="${YEARS:-1994 1995 1996 1997 1998 1999 2000 2001 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0
for Y in $YEARS; do
JOB="samudra-ufs-cplft7-trainclim-yr${Y}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "Training-period drift climatology hindcasts, year ${Y}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority high --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-trainclim \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/wave1_eval_configs/ufs-era5-cplft7-trainclim/yr${Y}.yaml" 2>&1)
echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "trainclim evals launched: $ok/$(echo $YEARS | wc -w)"
