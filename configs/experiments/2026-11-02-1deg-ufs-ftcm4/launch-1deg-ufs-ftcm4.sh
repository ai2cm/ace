#!/bin/bash
# UFS 1deg fine-tunes of the 1deg CM4 hybrid-residual + diagnosed-SST ppad pretrain (configs from make_configs.py),
# resources as the 1deg pretrain (4 GPUs, 747 GiB shared memory). The augmented UFS store lives on weka.
#   ARMS="resid-ohc-ppad-sstdiag-fmean-ufsft resid-noohc-ppad-sstdiag-fmean-ufsft" ./launch-1deg-ufs-ftcm4.sh
set -euo pipefail
MIN_RUNTIME="${MIN_RUNTIME:-6h}"   # beaker minimum guaranteed runtime (team template: 1deg 6h)
ARMS="${ARMS:-resid-ohc-ppad-sstdiag-fmean-ufsft resid-noohc-ppad-sstdiag-fmean-ufsft}"
CKPT_DS="${CKPT_DS:-01M626JPB247BQWSE0GN35SAXR}"   # samudra-ppad1deg-resid-ohc-ppad-sstdiag-fmean results (wandb jrgclj2f)
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
STATS_DS="${STATS_DS:?1deg UFS stats + CM4 placeholder entries dataset}"   # ufs-replay-ocean-1deg-19level-5day-stats-2026-10-06-cm4fill
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
JOB="samudra-ufs1deg-${A}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS 1deg replay: ${A} fine-tuned from the 1deg CM4 hybrid-residual diagnosed-SST ppad pretrain (${CKPT_DS})" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  $(for C in $CLUSTERS; do printf -- "--cluster %s " "$C"; done) \
  --weka climate-default:/climate-default \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs1deg-ftcm4 \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${STATS_DS}:/ocean_stats" \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --gpus "$N_GPUS" --shared-memory 747GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- torchrun --nproc_per_node="${N_GPUS}" -m fme.ace.train "${SCRIPT_PATH}/${A}.yaml" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')"; } || echo "FAILED $JOB: $(echo "$out" | tail -3)"
done
echo "1deg UFS fine-tunes launched: $ok/$(echo $ARMS | wc -w)"
