#!/bin/bash
# 1deg ACE-only random-CO2 evaluations (eval-ace/, held-out member ic_0003 at 1x, 2x and 4x CO2) of one atmosphere checkpoint:
# the standalone ACE scored with prescribed SST and sea ice, the same way Spencer scores the random-CO2-trained ACE2S arms.
# Used for "ACE before vs after coupled fine-tuning" (Troy 2026-10-15): score the atmosphere that goes into a coupled
# fine-tune, then the atmosphere lifted out of the coupled checkpoint (tools/extract_atmosphere_from_coupled_ckpt.py).
#   CKPT_DS=01M4Z3CQXD8BF13S4X6462XVNH TAG=rco2rs2-before ./launch-4deg-ace-randco2-evals.sh
#   CKPT_DS=<lifted atmosphere dataset> TAG=rco2rs2-after-jointens-ep<N> ./launch-4deg-ace-randco2-evals.sh
set -euo pipefail
CKPT_DS="${CKPT_DS:?Beaker dataset holding the atmosphere checkpoint}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
TAG="${TAG:?short label for the atmosphere, e.g. rco2rs2-before}"
CO2S="${CO2S:-1 2 4}"
PRIORITY="${PRIORITY:-high}"
MIN_RUNTIME="${MIN_RUNTIME:-1h}"   # guaranteed time before Beaker may preempt (gantry --min-runtime; 0 = any time)
CLUSTERS="${CLUSTERS:-ai2/ceres ai2/jupiter ai2/titan}"   # 1-GPU 4deg jobs: widen when the climate clusters are full
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for X in $CO2S; do
  total=$((total+1))
  JOB="ace1deg-randco2-${X}xCO2-ic3-${TAG}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "ACE-only random-CO2 ${X}xCO2 ic_0003 5.25-yr rollout of atmosphere ${TAG} (${CKPT_DS}:${CKPT_FILE})" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=ace1deg-randco2-before-after-coupled-ft \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval-ace/evaluator-config-randco2-${X}xCO2-ic3.yaml" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')"; } || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "ACE random-CO2 evals launched: $ok/$total"
