#!/bin/bash
# ENSO-by-coupled-stage follow-ups (elynn 2026-10-29), two sets:
#   ftalt  - FTO->FTalt training of the hybrid-residual mean-fill ocean (coupled/...-jointens-spb-fto-then-ftalt.yaml):
#            ocean from its FTO's lifted selected checkpoint, atmosphere from the 4deg ACE2S checkpoint.
#   fta    - 200-yr piControl / 130-yr 1pctCO2 coupled centennials of the three FTO->FTA runs' best_inference_ckpt,
#            named like the other stages' centennials (launch-4deg-ppad-coupled-ckpt-evals.sh), monthly files off.
#   DRY_RUN=1 ./launch-4deg-enso-stage-runs.sh ftalt fta
set -euo pipefail
WANDB_USERNAME="${WANDB_USERNAME:-elynnwu}"
[ "$WANDB_USERNAME" = elynnwu ] || { echo "WANDB_USERNAME must be elynnwu, got $WANDB_USERNAME"; exit 1; }
PRIORITY="${PRIORITY:-high}"
DRY="${DRY_RUN:+--dry-run}"
COUPLED_STATS=01KY8D816E67NXHP2EB29FHBJM   # 4deg cm4 coupled stats: coupled_atmosphere/ + ocean/
ATMOS_DS=01M0YFRM5TKQR6GSNDXKJSBNAS        # 4deg atmos_ace2s-ft rs0 results
HYBRID_FTO_OCEAN=01M508MVK5B0RVKG68E1W6VA3D   # hybrid FTO's selected checkpoint, ocean lifted (as for -fto-then-coupledft)
fta_ds() {   # results of troya/samudra-ppad4deg-<arm>-coupled-ft-fta-fto-then-fta
  case "$1" in
    ff-ohc-ppad) echo 01M554PBMJCQSE4X2TSNANBNM6 ;;
    ff-ohc-ppad-fmean) echo 01M5C3CVFHVVQYJFNJEHNZMQ6J ;;
    resid-ohc-cap0005-ppad-fmean) echo 01M55N763H8MRRNKC07K3EJE58 ;;
  esac
}
FTA_ARMS="ff-ohc-ppad ff-ohc-ppad-fmean resid-ohc-cap0005-ppad-fmean"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
[ -n "$SCRIPT_PATH" ] || { echo "empty SCRIPT_PATH"; exit 1; }
cd "$REPO_ROOT"
COMMON=(--beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")"
  --workspace ai2/ace --priority "$PRIORITY" --cluster ai2/jupiter --cluster ai2/titan
  --weka climate-default:/climate-default --env PYTORCH_ALLOC_CONF=expandable_segments:True
  --env WANDB_USERNAME="$WANDB_USERNAME" --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa
  --budget ai2/atec-climate --system-python --install "pip install --no-deps .")

launch_ftalt() {
  local A=resid-ohc-cap0005-ppad-fmean
  local JOB="samudra-ppad4deg-${A}-coupled-ft-jointens-spb-fto-then-ftalt"
  local CFG="${SCRIPT_PATH}/coupled/${A}-coupled-ft-jointens-spb-fto-then-ftalt.yaml"
  [ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }
  gantry run $DRY --name "$JOB" --task-name "$JOB" "${COMMON[@]}" \
    --description "4deg ppad ${A}: FTO then alternating coupled FT (3 ep, lr 1e-5)" \
    --min-runtime 3h --env FME_COLLECTIVE_TIMEOUT_MINUTES=120 \
    --env WANDB_NAME="$JOB" --env WANDB_JOB_TYPE=training --env WANDB_RUN_GROUP=samudra-ppad-4deg-coupled-ft \
    --dataset "${COUPLED_STATS}:coupled_atmosphere:/atmos_stats" --dataset "${COUPLED_STATS}:ocean:/ocean_stats" \
    --dataset "${HYBRID_FTO_OCEAN}:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar" \
    --dataset "${ATMOS_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar" \
    --gpus 2 --shared-memory 200GiB \
    -- torchrun --nproc_per_node=2 -m fme.coupled.train "$CFG"
}

launch_fta() {
  local A S CFG TAG JOB
  for A in $FTA_ARMS; do
    for S in piC 1pct; do
      case "$S" in
        piC)  CFG=coupled-evaluator-config-piC-ic0151-200yr-4deg-singleckpt.yaml;  TAG=piC-ic0151-200yr ;;
        1pct) CFG=coupled-evaluator-config-1pct-ic0001-130yr-4deg-singleckpt.yaml; TAG=1pct-ic0001-130yr ;;
      esac
      JOB="samudra-ppad4deg-${A}-coupled-ft-fta-fto-then-fta-eval-${TAG}"
      gantry run $DRY --name "$JOB" --task-name "$JOB" "${COMMON[@]}" \
        --description "4deg ppad ${A} FTO-then-FTA selected checkpoint: ${TAG} coupled free rollout" \
        --min-runtime 8h \
        --env WANDB_NAME="$JOB" --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ppad-4deg-centennial \
        --dataset "$(fta_ds "$A"):training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
        --gpus 1 --shared-memory 100GiB \
        -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" \
          --override data_writer.ocean.save_monthly_files=false
    done
  done
}

for set in "$@"; do
  case "$set" in
    ftalt) launch_ftalt ;;
    fta) launch_fta ;;
    *) echo "unknown set $set (ftalt|fta)"; exit 1 ;;
  esac
done
