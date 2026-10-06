#!/bin/bash
# 130-yr 1pctCO2 coupled rollouts in which the ocean reads upward shortwave (USWRFsfc) from the CM4
# forcing data instead of from ACE (CoupledStepperConfig.ocean_forcings_from_data, patched into the
# fine-tuned coupled checkpoints), everything else free-running. Tests whether the ocean's out-of-sample
# sea ice improves when the shortwave it gets over ice is right.
#   ./launch-4deg-coupled-uswrfdata-evals.sh
set -euo pipefail
PRIORITY="${PRIORITY:-high}"
declare -A CKPT=([ff-ohc-stoch-coupled-ft-ens]=01M496CVVRB95TPACAB9JT7CS3 [resid-ohc-coupled-ft]=01M496J0TK06AG107611GDFPM9)
SCENARIOS="${SCENARIOS:-1pct}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for A in "${!CKPT[@]}"; do
  for S in $SCENARIOS; do
    total=$((total+1))
    case "$S" in
      1pct) CFG=coupled-evaluator-config-1pct-ic0001-130yr-4deg-singleckpt.yaml; TAG=1pct-ic0001-130yr ;;
      piC)  CFG=coupled-evaluator-config-piC-ic0151-200yr-4deg-singleckpt.yaml; TAG=piC-ic0151-200yr ;;
    esac
    JOB="samudra-anchor4deg-${A}-uswrfdata-coupled-eval-${TAG}"
    out=$(gantry run --name "$JOB" --task-name "$JOB" \
      --description "4deg ${A}: coupled ${TAG} with USWRFsfc read from CM4 data (ocean_forcings_from_data)" \
      --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
      --workspace ai2/ace --priority "$PRIORITY" --preemptible \
      --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
      --weka climate-default:/climate-default \
      --env PYTORCH_ALLOC_CONF=expandable_segments:True \
      --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
      --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled-uswrfdata \
      --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
      --dataset "${CKPT[$A]}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
      --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
      --allow-dirty --system-python --install "pip install --no-deps ." \
      -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" 2>&1)
    echo "$out" | grep -qm1 "beaker.org/ex/" && ok=$((ok+1)) || echo "FAILED $JOB: $(echo "$out" | tail -2)"
  done
done
echo "uswrfdata coupled evals launched: $ok/$total"
