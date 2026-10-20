#!/bin/bash
# Generic ACE-only evaluation of one atmosphere checkpoint with one eval-ace/ config (the random-CO2 launcher's sibling for
# other scenarios, e.g. the 140-yr 1pctCO2 ramp). Job + wandb name: ace4deg-<CFG>-<TAG>, wandb project from the config,
# group ace4deg-ace-only-evals.
#   CFG=1pct-ic0001-140yr CKPT_DS=01M4Z3CQXD8BF13S4X6462XVNH TAG=rco2rs2-before ./launch-4deg-ace-eval.sh
set -euo pipefail
CFG="${CFG:?config name: eval-ace/evaluator-config-<CFG>.yaml}"
CKPT_DS="${CKPT_DS:?Beaker dataset holding the atmosphere checkpoint}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
TAG="${TAG:?short label for the atmosphere}"
PRIORITY="${PRIORITY:-high}"
MIN_RUNTIME="${MIN_RUNTIME:-}"   # e.g. 3h: ask Beaker not to preempt the job before this runtime (long evaluator rollouts restart from scratch when preempted)
CLUSTERS="${CLUSTERS:-ai2/ceres ai2/jupiter ai2/titan ai2/rhea ai2/prometheus ai2/saturn ai2/neptune}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
JOB="ace4deg-${CFG}-${TAG}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "ACE-only ${CFG} rollout of atmosphere ${TAG} (${CKPT_DS}:${CKPT_FILE})" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" ${MIN_RUNTIME:+--min-runtime "$MIN_RUNTIME"} ${MIN_RUNTIME:---preemptible} \
  $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=ace4deg-ace-only-evals \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/eval-ace/evaluator-config-${CFG}.yaml" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -2)"; exit 1; }
