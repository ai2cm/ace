#!/bin/bash
# Noise-draw ensemble hindcasts: rerun the 99-IC coupled protocol with fresh
# atmosphere-noise seeds. SYSTEM selects the checkpoint; existing unseeded
# runs are member 0, seeds 101-107 add 7 more members (8 total).
set -euo pipefail
SYSTEM="${SYSTEM:?cplft7 or julyft}"
case "$SYSTEM" in
  cplft7) DS=01M2GMQC7S44NCJN3NCS3CFEVV; FILE=best_inference_ckpt.tar;;
  julyft) DS=01KXFPKC7EMJ7G3VNDYFB4FZAX; FILE=training_checkpoints/best_inference_ckpt.tar;;
  *) echo "unknown SYSTEM"; exit 1;;
esac
SEEDS="${SEEDS:-101 102 103 104 105 106 107}"
YEARS="${YEARS:-2002 2003 2004 2005 2006 2007 2008 2009 2010}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0
for S in $SEEDS; do for Y in $YEARS; do
JOB="samudra-ufs-${SYSTEM}ens-s${S}-yr${Y}"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "ensemble member seed ${S}, ${SYSTEM}, year ${Y}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority high --preemptible \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-ufs-ens \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --dataset "${DS}:${FILE}:/ckpt.tar" \
  --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/wave1_eval_configs/ufs-era5-coupledft-ens/yr${Y}-s${S}.yaml" 2>&1)
echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out"|tail -1)"
done; done
echo "$SYSTEM ensemble launched: $ok/63"
