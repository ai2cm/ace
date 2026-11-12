#!/bin/bash
# Coupled evaluator of a 1deg UFS + ERA5 coupled fine-tune checkpoint (single-checkpoint form; configs in ./eval).
#   CKPT_DS=<results ds> NAME=<label> CFG=coupled-evaluator-config-ufs1deg-era5-fullrecord-1994-singleckpt.yaml ./launch-1deg-ufs-era5-coupled-eval.sh
#   CKPT_DS=<results ds> NAME=<label> HINDCAST_YEARS="2002 ... 2011" ./launch-1deg-ufs-era5-coupled-eval.sh   # one job per start year
#   ... CFG_SUFFIX=-nocorr ...   # hindcasts with the ocean heat-content correction removed at inference (corrector-kept ckpts)
set -euo pipefail
CKPT_DS="${CKPT_DS:?coupled fine-tune results dataset}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
NAME="${NAME:?job label, e.g. ace2s6h-fto-then-ftalt}"
FRAC_DS="${FRAC_DS:?surface-fraction store dataset}"
FRAC_ZARR=2026-11-05-era5-1deg-6h-ufs-coupled-fractions-1994-2023.zarr
CFG="${CFG:-coupled-evaluator-config-ufs1deg-era5-fullrecord-1994-singleckpt.yaml}"
HINDCAST_YEARS="${HINDCAST_YEARS:-}"
CFG_SUFFIX="${CFG_SUFFIX:-}"   # e.g. -nocorr: hindcast configs coupled-evaluator-config-...-hindcast-<Y>-singleckpt${CFG_SUFFIX}.yaml
MIN_RUNTIME="${MIN_RUNTIME:-6h}"
PRIORITY="${PRIORITY:-high}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
launch() {  # job name, config file
  local JOB=$1 C=$2
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "1deg UFS + ERA5 coupled evaluator (${C}) of ${CKPT_DS}:${CKPT_FILE}" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
    --weka climate-default:/climate-default \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-ufs1deg-era5-coupled-eval \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --dataset "${FRAC_DS}:/era5frac/${FRAC_ZARR}" \
    --gpus 1 --shared-memory 200GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${C}" 2>&1) || true
  echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; return 1; }
}
if [ -n "$HINDCAST_YEARS" ]; then
  for Y in $HINDCAST_YEARS; do launch "samudra-ufs1deg-era5-${NAME}-hindcast-${Y}" "coupled-evaluator-config-ufs1deg-era5-hindcast-${Y}-singleckpt${CFG_SUFFIX}.yaml"; done
else
  launch "samudra-ufs1deg-era5-${NAME}-fullrecord-1994" "$CFG"
fi
