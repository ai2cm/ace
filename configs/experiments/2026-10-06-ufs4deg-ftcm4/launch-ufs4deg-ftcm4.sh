#!/bin/bash
# Deterministic 4deg hybrid-residual oceans fine-tuned from their CM4 checkpoints onto the rebuilt UFS replay
# 4deg 5-day store (CM4 level interfaces), to diagnose the unaccounted ocean heating of the replay data.
# Each arm's weights come from the CM4 pretrain of the same name (best_inference_ckpt); the UFS store is the
# Beaker dataset DATA_DS (the GCS store plus hfgeou from CM4 and zero sfdsi/UI/VI so the CM4 variable set loads).
#   ARMS="resid-ohc resid-ohc-cap0005 resid-noohc" ./launch-ufs4deg-ftcm4.sh
set -euo pipefail
ARMS="${ARMS:-resid-ohc resid-ohc-cap0005 resid-noohc}"
DATA_DS="${DATA_DS:?set DATA_DS to the Beaker dataset id of ufs-replay-ocean-4deg-19level-5day-2026-10-02-cm4vars}"
STATS_DS="${STATS_DS:-01KY8D816E67NXHP2EB29FHBJM}"   # CM4 4deg stats: the checkpoints' own normalization
PRIORITY="${PRIORITY:-high}"
N_GPUS="${N_GPUS:-1}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  total=$((total+1))
  case "$A" in
    resid-noohc) SRC=resid-ohc ;;   # no CM4 pretrain without the corrector is stable; start from the scaled arm's weights
    *) SRC=$A ;;
  esac
  CKPT_DS=$(beaker experiment get "troya/samudra-anchor4deg-${SRC}" --format json | jq -r '.[0].jobs[-1].result.beaker')
  [ -n "$CKPT_DS" ] && [ "$CKPT_DS" != "null" ] || { echo "no results dataset for $SRC"; continue; }
  JOB="samudra-ufs4deg-${A}-ftcm4"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "UFS 4deg replay: ${A} fine-tuned from CM4 ${SRC} best_inference_ckpt" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs4deg-ftcm4 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${STATS_DS}:ocean:/ocean_stats" \
    --dataset "${DATA_DS}:/ufs4deg/2026-10-02-ufs-replay-ocean-4deg-19level-1994-2023.zarr" \
    --dataset "${CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "${SCRIPT_PATH}/${A}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "UFS 4deg fine-tunes launched: $ok/$total"
