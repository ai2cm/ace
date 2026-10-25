#!/bin/bash
# Launch a coupled SamudrACE forecast from an initial condition built by make_ic.py.
#
#   IC_DIR=/path/to/ic DATE=2026-10-16 ./launch-operational-forecast.sh
#
# Uploads the IC directory as a Beaker dataset (unless IC_DS is given), mounts
# it at /ic together with the deployed checkpoint, rewrites the local paths of
# inference_config.yaml to the container paths, and runs
# fme.coupled.inference on one GPU with the team's --min-runtime template
# (never --preemptible: minRuntime 0 sits unscheduled for hours).
# DRY_RUN=1 skips the dataset upload and passes --dry-run to gantry.
set -euo pipefail
DRY_RUN="${DRY_RUN:-0}"
IC_DIR="${IC_DIR:?set IC_DIR to the make_ic.py output directory}"
DATE="${DATE:?set DATE to the issue day (YYYY-MM-DD)}"
CKPT_DS="${CKPT_DS:-01M5N1GVAVA2GEEPVDQ8VPT98W}"   # samudra-glo12-era5-coupledft-spb results
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt_0007.tar}"   # epoch 7 (or 9): best on the 2025 hindcasts
IC_DS="${IC_DS:-}"
NAME="${NAME:-forecast-${DATE}}"
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
PRIORITY="${PRIORITY:-high}"
WANDB="${WANDB:-true}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
cd "$REPO_ROOT"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
JOB="samudra-glo12-era5-operational-${NAME}"

# the config with container paths (the IC dataset is mounted at /ic)
CFG="${SCRIPT_PATH}/configs/inference-${NAME}.yaml"
mkdir -p "${SCRIPT_PATH}/configs"
sed -e "s#${IC_DIR%/}#/ic#g" -e "s#experiment_dir: .*#experiment_dir: /results#" \
    -e "s#log_to_wandb: .*#log_to_wandb: ${WANDB}#" "${IC_DIR}/inference_config.yaml" > "$CFG"
grep -q "/ic/" "$CFG" || { echo "config rewrite failed: no /ic/ paths in $CFG"; exit 1; }

if [ "$DRY_RUN" = "1" ]; then
  IC_DS="${IC_DS:-00000000000000000000000000}"
  echo "dry run: config $CFG, IC dataset placeholder $IC_DS"
elif [ -z "$IC_DS" ]; then
  IC_DS_NAME="samudrace-operational-ic-${DATE}-$(date -u +%Y%m%dT%H%M%S)"
  beaker dataset create --workspace ai2/ace --name "$IC_DS_NAME" "$IC_DIR" > /dev/null
  # named datasets resolve as <author>/<name>; create prints no id when piped
  IC_DS=$(beaker dataset get "${BEAKER_USERNAME}/${IC_DS_NAME}" --format json \
    | jq -r 'if type == "array" then .[0].id else .id end')
  n_files=$(beaker dataset ls "$IC_DS" --format json | jq 'length')
  [ "$n_files" -gt 0 ] || { echo "dataset $IC_DS is empty"; exit 1; }
  echo "IC dataset $IC_DS ($n_files files)"
fi

out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "SamudrACE GLO12+ERA5 operational forecast from ${DATE}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" \
  --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$JOB" \
  --env WANDB_JOB_TYPE=inference --env WANDB_RUN_GROUP=samudra-glo12-era5-operational \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset "${CKPT_DS}:${CKPT_FILE}:/ckpt.tar" \
  --dataset "${IC_DS}:/ic" \
  --gpus 1 --shared-memory 100GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  $([ "$DRY_RUN" = "1" ] && echo --dry-run) \
  -- python -I -m fme.coupled.inference "$CFG" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB" || { echo "FAILED $JOB: $(echo "$out" | tail -5)"; exit 1; }
