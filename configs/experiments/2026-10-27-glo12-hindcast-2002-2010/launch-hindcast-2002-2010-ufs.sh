#!/bin/bash
# UFS-trained coupled SamudrACE on its own 2002-2010 holdout from UFS replay ICs, one Beaker job per year (defaults set; just run it).
#   OCEAN_DS=<derived GLORYS12 store ds> ERA5_DS=<ERA5 GLORYS-period store ds> YEARS="2002 2003" ./launch-hindcast-2002-2010.sh
set -euo pipefail
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
CKPT_DS="${CKPT_DS:-01M5DXQST7WCYTXYXKMEF3KQ13}"   # UFS-trained alternating coupled FT (spb-ufsstats), best ep12
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
OCEAN_DS="${OCEAN_DS:-01M4E83P5GMXQEN2JN8SZFB5BN}"   # ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh
ERA5_DS="${ERA5_DS:-01M4W630AM1HQRYKJ2AJ0JMNN1}"   # era5-4deg-8layer-daily-ufs-coupled-1994-2023-v2
OCEAN_ZARR=2026-10-06-ufs-replay-ocean-4deg-19level-1994-2023.zarr
ERA5_ZARR=2026-10-14-era5-4deg-8layer-daily-ufs-coupled-1994-2023.zarr
YEARS="${YEARS:-2002 2003 2004 2005 2006 2007 2008 2009 2010}"
PRIORITY="${PRIORITY:-high}"
TAG="${TAG:-ufsspb}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
for Y in $YEARS; do
  JOB="samudra-glo12-era5-coupled-hindcast0210-${TAG}-${Y}"
  CFG="coupled-evaluator-config-ufs-era5-hindcast-${Y}.yaml"
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "UFS-trained coupled SamudrACE (${TAG}) 2002-2010 S2S hindcast on UFS replay ICs: ${Y}, 12 starts x 8 members x 90 d" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-glo12-hindcast-2002-2010 \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
    --dataset "${OCEAN_DS}:/ufs4deg/${OCEAN_ZARR}" \
    --dataset "${ERA5_DS}:/era5/${ERA5_ZARR}" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/${CFG}" 2>&1) || true
  if echo "$out" | grep -qm1 "beaker.org/ex/"; then
    echo "launched $JOB $(echo "$out" | grep -om1 'beaker.org/ex/[A-Z0-9]*')"
  else
    echo "FAILED $JOB: $(echo "$out" | tail -3)"
  fi
done
