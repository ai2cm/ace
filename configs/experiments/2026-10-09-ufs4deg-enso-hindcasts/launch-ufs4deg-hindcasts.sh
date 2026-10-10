#!/bin/bash
# ENSO hindcasts (120 monthly ICs, 2002-2011 holdout, 128 steps, ocean-only) for the 4deg UFS arms and the CM4 anchors.
#   MODE=true|clim ARMS="scratch-ff-ohc ..." ./launch-ufs4deg-hindcasts.sh
# MODE=true mounts the training store (cm4vars-uh superset, true replay forcing); MODE=clim mounts the
# climatological-forcing copy (CLIM_DS). Arm -> checkpoint results dataset in ARM_CKPT below (best_inference_ckpt).
set -euo pipefail
MODE="${MODE:-true}"
TRUE_DS="${TRUE_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"   # ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh
CLIM_DS="${CLIM_DS:-01M4HGTKEM1ZA770N6XYKBBKTG}"   # ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh-climforcing
PRIORITY="${PRIORITY:-high}"
declare -A ARM_CKPT=(
  [scratch-ff-ohc]=01M4GG3NPE68TAHMASSQDP7KN6 [scratch-resid-ohc-cap0005]=01M4GAA5S0HZHCH050GTT1E9SP [scratch-resid-noohc]=01M4FXFZEQ4E87VFMQZM1Q8QJ8
  [ftcm4-resid-ohc]=01M4E7NG96SV956P6HF9FMW02H [ftcm4-resid-ohc-cap0005]=01M4E7NMXAEZY6W7M442G5AF24 [ftcm4-resid-noohc]=01M4E7NS1M93SJMWBXPC8QNCM3
  [cm4zs-ff-ohc]=01M2XA1DZ91Q2W0Y3YA3R5J4GT [cm4zs-resid-ohc]=01M2TYRTSSTQKCYNT6TY0WYBTA [cm4zs-resid-ohc-cap0005]=01M3W3FKM09M13ZN021567MW6Q
  [scratch-resid-cap0005-learnedheat]=01M4GCJRDQDBGTF9TK4KKM64QY [scratch-ff-ohc-learnedheat]=01M4GG3NT1QA1RHV7N9A4A86NX
)
ARMS="${ARMS:-${!ARM_CKPT[@]}}"
ZARR=2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr
case "$MODE" in
  true) DATA_MOUNT="${TRUE_DS}:/ufs4deg/${ZARR}" ;;                 # zarr contents at the dataset root
  clim) DATA_MOUNT="${CLIM_DS:?CLIM_DS required for MODE=clim}:${ZARR}:/ufs4deg/${ZARR}" ;;   # zarr is a subdirectory
  *) echo "MODE must be true or clim"; exit 1 ;;
esac
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in $ARMS; do
  total=$((total+1)); CKPT_DS=${ARM_CKPT[$A]}
  # the CM4 checkpoints carry CM4's wet mask in their dataset info; the UFS store's differs, so skip that check
  case "$A" in cm4zs-*) OVERRIDE="allow_incompatible_dataset=true" ;; *) OVERRIDE="" ;; esac
  JOB="samudra-ufs4deg-hindcast-${A}-${MODE}forcing"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "UFS 4deg ENSO hindcasts, ${A}, ${MODE} forcing: 120 monthly ICs 2002-2011 x 128 steps, ocean-only" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs4deg-hindcasts \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
    --dataset "${DATA_MOUNT}" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.ace.evaluator "${SCRIPT_PATH}/evaluator-config-ufs4deg-hindcast-2002-2011.yaml" ${OVERRIDE:+--override "$OVERRIDE"} 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB"; } || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "hindcasts launched: $ok/$total"
