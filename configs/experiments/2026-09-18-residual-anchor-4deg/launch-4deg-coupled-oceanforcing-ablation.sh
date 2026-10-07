#!/bin/bash
# Ocean-forcing ablation for coupled sea ice: 50-yr 1pctCO2 coupled rollouts of the ff-ohc-stoch
# coupled-fine-tuned (ensemble) 4deg checkpoint in which chosen atmosphere-produced ocean forcings
# are read from the CM4 data (InferenceEvaluatorConfig.ocean_forcings_from_data), everything else
# free-running. "all" is the Samudra-only control (every coupled flux prescribed); "free" is the
# plain coupled run. Years 1-40 are outside the 1pct training window.
#   ./launch-4deg-coupled-oceanforcing-ablation.sh            # all 16 variants
#   VARIANTS="all free rad" ./launch-4deg-coupled-oceanforcing-ablation.sh
#   FORM=two OCEAN_DS=<pretrain results> VARIANTS="free all" ./launch-4deg-coupled-oceanforcing-ablation.sh   # ocean before coupled FT
set -euo pipefail
PRIORITY="${PRIORITY:-high}"
CKPT_DS="${CKPT_DS:-01M3W6QQ0HTJF39CJ8MK5GQW2S}"   # samudra-anchor4deg-ff-ohc-stoch-coupled-ft-ens results
FORM="${FORM:-single}"                              # single: one coupled checkpoint; two: ocean pretrain + frozen atmosphere
OCEAN_DS="${OCEAN_DS:-01M3PZ8STAHR0MGF3D4ZQCR6ZP}"  # samudra-anchor4deg-ff-ohc-stoch (pretrain) results, FORM=two
ATMOS_DS="${ATMOS_DS:-01M0YFRM5TKQR6GSNDXKJSBNAS}"  # 4deg atmos_ace2s-ft rs0 results, FORM=two
NAME_TAG="${NAME_TAG:-ffstoch-cft-ens}"
if [ "$FORM" = "two" ]; then
  CFG=coupled-evaluator-config-1pct-ic0001-50yr-4deg-twockpt-oceanforcing.yaml
  OVERRIDE_KEY=checkpoint_path.ocean_forcings_from_data
  NAME_TAG="${NAME_TAG_TWO:-ffstoch-noft}"
else
  CFG=coupled-evaluator-config-1pct-ic0001-50yr-4deg-singleckpt-oceanforcing.yaml
  OVERRIDE_KEY=ocean_forcings_from_data
fi
declare -A FORCINGS=(
  [free]=""
  [all]="DLWRFsfc,DSWRFsfc,ULWRFsfc,USWRFsfc,LHTFLsfc,SHTFLsfc,PRATEsfc,total_frozen_precipitation_rate,eastward_surface_wind_stress,northward_surface_wind_stress"
  [rad]="DLWRFsfc,DSWRFsfc,ULWRFsfc,USWRFsfc"
  [turb]="LHTFLsfc,SHTFLsfc"
  [water]="PRATEsfc,total_frozen_precipitation_rate"
  [stress]="eastward_surface_wind_stress,northward_surface_wind_stress"
  [dlwrf]="DLWRFsfc" [dswrf]="DSWRFsfc" [ulwrf]="ULWRFsfc" [uswrf]="USWRFsfc"
  [lhtfl]="LHTFLsfc" [shtfl]="SHTFLsfc" [prate]="PRATEsfc" [frozen]="total_frozen_precipitation_rate"
  [taux]="eastward_surface_wind_stress" [tauy]="northward_surface_wind_stress"
)
VARIANTS="${VARIANTS:-free all rad turb water stress dlwrf dswrf ulwrf uswrf lhtfl shtfl prate frozen taux tauy}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"
ok=0; total=0
for V in $VARIANTS; do
  total=$((total+1))
  JOB="samudra-anchor4deg-${NAME_TAG}-oforce-${V}-1pct-ic0001-50yr"
  OVERRIDE=()
  if [ -n "${FORCINGS[$V]}" ]; then OVERRIDE=(--override "${OVERRIDE_KEY}=[${FORCINGS[$V]}]"); fi
  if [ "$FORM" = "two" ]; then
    MOUNTS=(--dataset "${OCEAN_DS}:training_checkpoints/best_inference_ckpt.tar:/ocean_ckpt.tar" --dataset "${ATMOS_DS}:training_checkpoints/best_inference_ckpt.tar:/atmos_ckpt.tar")
  else
    MOUNTS=(--dataset "${CKPT_DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar")
  fi
  out=$(gantry run --name "$JOB" --task-name "$JOB" \
    --description "4deg ${NAME_TAG}: 50-yr 1pct with ocean forcings [${FORCINGS[$V]:-none}] read from CM4 data" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env PYTORCH_ALLOC_CONF=expandable_segments:True \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
    --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-anchor-4deg-coupled-oceanforcing-ablation \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    "${MOUNTS[@]}" \
    --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- python -I -m fme.coupled.evaluator "${SCRIPT_PATH}/eval/${CFG}" "${OVERRIDE[@]}" 2>&1)
  echo "$out" | grep -qm1 "beaker.org/ex/" && { ok=$((ok+1)); echo "launched $JOB"; } || echo "FAILED $JOB: $(echo "$out" | tail -2)"
done
echo "ocean-forcing ablation launched: $ok/$total"
