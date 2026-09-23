#!/bin/bash
# 1deg coupled fine-tunes of the piControl+1pctCO2 hybrid-residual ocean, on the team's recipe
# (see make_coupled_ft_config_1deg.py). Mounts mirror the team's coupled jobs exactly.
#   VARIANTS="ens mse" ./launch-coupled-ft-1deg.sh
set -euo pipefail
VARIANTS="${VARIANTS:-ens mse}"
PRIORITY="${PRIORITY:-high}"
STATS_DS="${STATS_DS:-01KXNT0RA6VX2YTZ8WJ936Q5RS}"      # 1deg CM4 coupled stats (ocean/, coupled_atmosphere/)
ATMOS_DS="${ATMOS_DS:-01M0VTG1SJQK69WWN5A651RXZY}"      # 1deg ACE2S atmosphere (a2s), best_inference_ckpt
OCEAN_DS="${OCEAN_DS:-01M2SA801AHDGQH8GMAK86E7X8}"      # elynn: ocean_wide_hybridresid_clip1_sstff_ohc-ft-rs0 (piC+1pct, 8-step FT)
N_GPUS="${N_GPUS:-4}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for V in $VARIANTS; do
  total=$((total+1))
  JOB="samudra-hybridresid-pic1pct-coupled-ft-${V}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg coupled FT of the piC+1pct hybrid-residual ocean, ${V} loss (team recipe)" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-coupled-ft-1deg \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${STATS_DS}:coupled_atmosphere:/atmos_stats" \
    --dataset "${STATS_DS}:ocean:/ocean_stats" \
    --dataset "${ATMOS_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar" \
    --dataset "${OCEAN_DS}:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar" \
    --gpus "$N_GPUS" --shared-memory 400GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${SCRIPT_PATH}/hybridresid-pic1pct-coupled-ft-${V}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "1deg coupled FTs launched: $ok/$total"
