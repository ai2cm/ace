#!/bin/bash
# 2002-2010 clean-window S2S hindcast of the GLO12 coupled SamudrACE checkpoint on GLORYS12 ICs, one Beaker job per year:
#   OCEAN_DS=<derived GLORYS12 store ds> ERA5_DS=<ERA5 GLORYS-period store ds> YEARS="2002 2003" ./launch-hindcast-2002-2010.sh
set -euo pipefail
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
CKPT_DS="${CKPT_DS:-01M5N1GVAVA2GEEPVDQ8VPT98W}"   # GLO12 coupled FT results (samudra-glo12-era5-coupledft-spb)
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt_0007.tar}"   # epoch 7 = the operational checkpoint
OCEAN_DS="${OCEAN_DS:?set OCEAN_DS}"
ERA5_DS="${ERA5_DS:?set ERA5_DS}"
OCEAN_ZARR=2026-10-27-glorys12-ocean-4deg-19level-5day-2001-2011-cm4vars.zarr
ERA5_ZARR=2026-10-27-era5-4deg-8layer-daily-glorys12-coupled-2001-2011.zarr
YEARS="${YEARS:-2002 2003 2004 2005 2006 2007 2008 2009 2010}"
PRIORITY="${PRIORITY:-high}"
TAG="${TAG:-ep7}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for Y in $YEARS; do
  JOB="samudra-glo12-era5-coupled-hindcast0210-${TAG}-${Y}"
  CFG="coupled-evaluator-config-glorys12-era5-hindcast-${Y}.yaml"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "GLO12 coupled SamudrACE (${TAG}) 2002-2010 S2S hindcast on GLORYS12 ICs: ${Y}, 12 starts x 8 members x 90 d" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-glo12-hindcast-2002-2010 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --dataset "${OCEAN_DS}:/glorys/${OCEAN_ZARR}" \
    --dataset "${ERA5_DS}:/era5glorys/${ERA5_ZARR}" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/${CFG}" 2>&1) || true
  if echo "$out" | grep -qm1 "beaker.org/ex/"; then
    echo "launched $JOB $(echo "$out" | grep -om1 'beaker.org/ex/[A-Z0-9]*')"
  else
    echo "FAILED $JOB: $(echo "$out" | tail -3)"
  fi
done
