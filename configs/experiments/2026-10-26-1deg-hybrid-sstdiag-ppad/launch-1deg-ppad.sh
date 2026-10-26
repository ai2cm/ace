#!/bin/bash
# 1deg ocean-only pretrain(s) on James's ocean_ppad baseline, resources as his rs0 runs (4 GPUs, 747 GiB shared memory, 8h min runtime).
#   ARMS="resid-ohc-ppad-sstdiag-fmean-1deg" ./launch-1deg-ppad.sh
set -euo pipefail
MIN_RUNTIME="${MIN_RUNTIME:-8h}"   # beaker minimum guaranteed runtime (James's 1deg pretrains: 8h)
STATS_DS="${STATS_DS:-01KXNT0RA6VX2YTZ8WJ936Q5RS}"   # 1deg cm4 coupled stats (ocean/: centering, scaling-full-field, scaling-residual)
ARMS="${ARMS:-resid-ohc-ppad-sstdiag-fmean-1deg}"
N_GPUS="${N_GPUS:-4}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/jupiter ai2/titan ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0
for A in $ARMS; do
JOB="samudra-ppad1deg-${A%-1deg}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg ocean_ppad pretrain, arm ${A}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ppad-1deg \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${STATS_DS}:ocean:/ocean_stats" \
  --gpus "$N_GPUS" --shared-memory 747GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.ace.train "${SCRIPT_PATH}/${A}.yaml" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')"; } || echo "FAILED $JOB: $(echo "$out" | tail -3)"
done
echo "1deg ppad arms launched: $ok/$(echo $ARMS | wc -w)"
