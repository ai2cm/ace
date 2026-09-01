#!/bin/bash
#
# Prescribe-from-truth probe: where in the coupled state does ENSO phase
# information get lost?
#
# The wind-stress bridge audit came back healthy (coupling gain ratio 0.98,
# 95% CI [0.89, 1.08]; interface noise 0.0144 vs 0.0148 N/m2), so the loss is
# somewhere the scalar bridge measurement cannot see. Each arm holds one part
# of the coupled state at CM4 truth during an otherwise free rollout; how much
# Nino3.4 skill returns says which part is being corrupted.
#
# Arms (see make_year_configs_prescribe.py):
#   free        nothing prescribed (baseline, rerun only to write the ocean
#               interior no previous eval saved)
#   subsurface  thetao_1..18 + zos
#   sst         sst
#   currents    uo_*/vo_*/ssu/ssv
#   windstress  atmosphere wind stress
#   fluxes      atmosphere surface fluxes + precip
#
# Same job layout as the existing evals: one gantry job per year, 12 monthly
# ICs, n_coupled_steps=146, N_GPUS=1. IC timestamps are the same ones the
# earlier AR-SST / AR-readout / one-step evals used, so the resulting skill
# curves are directly comparable to the free-run and MLP-readout numbers
# without rescoring anything.
#
# Default years 0233/0246/0250 are the three most ENSO-active years of the
# 0231-0250 window (index sd 1.07 / 0.87 / 1.00), chosen so a skill difference
# between arms is visible with only 36 ICs.
#
# Examples:
#   ARMS=subsurface YEARS=233 ./evaluate-prescribe-arms.sh     # smoke test
#   ./evaluate-prescribe-arms.sh                                # all arms
#   DRY_RUN=1 ./evaluate-prescribe-arms.sh                      # print only

set -euo pipefail

JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino-prescribe}"
COUPLED_RESULTS_DATASET="${COUPLED_RESULTS_DATASET:-01KY3DATM3CAEA479JQZQDPT9W}"
COUPLED_CKPT="${COUPLED_CKPT:-best_inference_ckpt}"
ARMS="${ARMS:-free subsurface sst currents windstress fluxes}"
YEARS="${YEARS:-233 246 250}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
CONFIG_DIR="${SCRIPT_PATH}/prescribe_year_configs"
MAKE_SCRIPT="${SCRIPT_PATH}/make_year_configs_prescribe.py"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
N_GPUS=1

cd "$REPO_ROOT"

n_jobs=$(( $(wc -w <<<"$ARMS") * $(wc -w <<<"$YEARS") ))
echo "prescribe probe: arms [$ARMS] x years [$YEARS] = ${n_jobs} jobs, ${N_GPUS} GPU each"

run_one() {
  local arm="$1" year="$2"
  local year_str
  year_str=$(printf "%04d" "$year")
  local config_path="${CONFIG_DIR}/${arm}/yr${year_str}.yaml"
  local job_name="cm4-coupled-ft-nino-prescribe-${arm}-yr${year_str}"

  # Generate and schema-check locally first so a bad arm fails here, not on a GPU.
  python "$MAKE_SCRIPT" --arm "$arm" --years "$year" >/dev/null
  python -m fme.coupled.validate_config --config_type evaluator "$config_path"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [dry-run] $job_name"
    return
  fi

  gantry run \
    --name "$job_name" \
    --task-name "$job_name" \
    --description "Prescribe-from-truth probe (${arm}): which part of the coupled state carries ENSO phase" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 2h \
    --cluster ai2/ceres \
    --cluster ai2/jupiter \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=evaluation \
    --env WANDB_RUN_GROUP="$JOB_GROUP" \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "$COUPLED_RESULTS_DATASET:training_checkpoints/${COUPLED_CKPT}.tar:/ckpt.tar" \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- bash -c \
        "python '$MAKE_SCRIPT' --arm $arm --years $year && \
         python -I -m fme.coupled.evaluator '$config_path'"
}

for arm in $ARMS; do
  for year in $YEARS; do
    run_one "$arm" "$year"
  done
done
