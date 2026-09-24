#!/bin/bash
# Train the bounded 4deg ocean arms FROM SCRATCH inside the coupled system (frozen stochastic
# 4deg ACE2S atmosphere), instead of pretraining with perfect forcing and then coupled fine-tuning.
# Same launcher as launch-4deg-coupled-ft.sh minus the ocean checkpoint mount.
#   ARMS="ff-ohc resid-ohc resid-anomohc" ./launch-4deg-coupled-scratch.sh
#   VARIANT=scratch-trueocean ./launch-4deg-coupled-scratch.sh   # atmosphere forced by the true ocean surface
set -euo pipefail
ARMS="${ARMS:-ff-ohc resid-ohc resid-anomohc}"
VARIANT="${VARIANT:-scratch}"   # scratch | scratch-trueocean (atmosphere forced by the true ocean surface in training)
COUPLED_STATS="${COUPLED_STATS:-01KY8D816E67NXHP2EB29FHBJM}"   # 4deg cm4 coupled stats: coupled_atmosphere/ + ocean/
ATMOS_DS="${ATMOS_DS:-01M0YFRM5TKQR6GSNDXKJSBNAS}"           # 4deg atmos_ace2s-ft rs0 results
ATMOS_CKPT="${ATMOS_CKPT:-training_checkpoints/best_inference_ckpt.tar}"
N_GPUS="${N_GPUS:-4}"
PRIORITY="${PRIORITY:-high}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  total=$((total+1))
  JOB="samudra-anchor4deg-${A}-coupled-${VARIANT}"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg anchor arm ${A}: ocean trained from scratch inside the coupled system (frozen stochastic ACE2S atmosphere), variant ${VARIANT}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled-${VARIANT} \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${COUPLED_STATS}:coupled_atmosphere:/atmos_stats" \
    --dataset "${COUPLED_STATS}:ocean:/ocean_stats" \
    --dataset "${ATMOS_DS}:${ATMOS_CKPT}:/atmos_ckpt.tar" \
    --gpus "$N_GPUS" --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node="${N_GPUS}" -m fme.coupled.train "${SCRIPT_PATH}/coupled/${A}-coupled-${VARIANT}.yaml" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "coupled from-scratch launched: $ok/$total"
