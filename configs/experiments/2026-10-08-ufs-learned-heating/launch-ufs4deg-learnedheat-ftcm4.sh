#!/bin/bash
# Learned unaccounted heating on the program path: CM4 4deg hybrid-residual checkpoint fine-tuned on the UFS 4deg replay store
# 4deg 5-day store (CM4 level interfaces), to diagnose the unaccounted ocean heating of the replay data.
# Each arm's weights come from the CM4 pretrain of the same name (best_inference_ckpt); the UFS store is the
# Beaker dataset DATA_DS (the GCS store plus hfgeou from CM4 and zero sfdsi/UI/VI so the CM4 variable set loads).
#   ./launch-ufs4deg-learnedheat-ftcm4.sh   (ARMS=... to pick; default stats = UFS + unaccounted_heating; the 2026-10-14 arms
#   ran with STATS_DS=01M4W3MH28S5HPAJ84R7AB4V3W STATS_SUBDIR=ocean, the CM4 stats, which is wrong for a UFS fine-tune)
set -euo pipefail
ARMS="${ARMS:-resid-ohc-cap0005-learnedheat-ftcm4 resid-ohc-cap0005-const0-ftcm4}"
DATA_DS="${DATA_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"   # cm4vars-uh store (CM4 variable set + unaccounted_heating target)
STATS_DS="${STATS_DS:-01M4E83B1B83CB8DV9YEQTGA4G}"   # UFS 4deg 2026-10-06 stats + unaccounted_heating stats (files at the dataset root)
STATS_SUBDIR="${STATS_SUBDIR:-}"   # "ocean" for the CM4 stats dataset (01M4W3MH28S5HPAJ84R7AB4V3W, the ORIGINAL arms; CM4 stats are for pretraining only, Troy 2026-10-20)
JOB_SUFFIX="${JOB_SUFFIX:-}"       # e.g. -ufsstats
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
    *-ftcm4) SRC=resid-ohc-cap0005 ;;   # both arms start from the CM4 hybrid-residual anchor
    *) SRC=$A ;;
  esac
  if [ -n "${CKPT_DS_OVERRIDE:-}" ]; then CKPT_DS=$CKPT_DS_OVERRIDE; else   # e.g. the ufsvars CM4 pretraining's results dataset
  CKPT_DS=$(beaker experiment get "troya/samudra-anchor4deg-${SRC}" --format json | jq -r '.[0].jobs[-1].result.beaker'); fi
  [ -n "$CKPT_DS" ] && [ "$CKPT_DS" != "null" ] || { echo "no results dataset for $SRC"; continue; }
  JOB="samudra-ufs4deg-${A}${JOB_SUFFIX}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "UFS 4deg replay: ${A} fine-tuned from CM4 ${SRC} best_inference_ckpt" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ufs4deg-learnedheat-ftcm4 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${STATS_DS}:${STATS_SUBDIR:+${STATS_SUBDIR}:}/ocean_stats" \
    --dataset "${DATA_DS}:/ufs4deg/2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr" \
    --dataset "${CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "${SCRIPT_PATH}/${A}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "UFS 4deg fine-tunes launched: $ok/$total"
