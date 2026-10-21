#!/bin/bash
# Ocean-only rollouts of a UFS 4deg checkpoint with monthly outputs: 10-yr from 2012/2013 (default) or the
# full record from 1994 (CFG=evaluator-config-ufs4deg-30yr-1994.yaml TAG=30yr-1994).
#   CKPT_DS=<results dataset> NAME=resid-cap0005-learnedheat [DATA_DS=<training dataset>] ./launch-ufs4deg-eval-10yr.sh
set -euo pipefail
CKPT_DS="${CKPT_DS:?results dataset holding training_checkpoints/best_inference_ckpt.tar}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME="${NAME:?arm name}"
DATA_DS="${DATA_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"   # …-cm4vars-uh (has unaccounted_heating for scoring; a superset of the clean store)
PRIORITY="${PRIORITY:-high}"
CFG="${CFG:-evaluator-config-ufs4deg-10yr-2012-2013.yaml}"
TAG="${TAG:-10yr-2012-2013}"
EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}"   # e.g. data_writer.names=[sst,thetao_0] for checkpoints without the learned channel
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="samudra-ufs4deg-${NAME}-eval-${TAG}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 4deg ${NAME}: 10-yr ocean-only rollouts from 2012 and 2013 with monthly outputs" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs4deg-eval-10yr \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --dataset "${DATA_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr" \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval/${CFG}" ${EXTRA_OVERRIDE:+--override "$EXTRA_OVERRIDE"} 2>&1)
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || echo "FAILED $JOB: $(echo "$out" | tail -2)"
