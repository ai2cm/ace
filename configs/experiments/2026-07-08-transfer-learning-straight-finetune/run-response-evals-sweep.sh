#!/bin/bash
#
# EPOCH SWEEP of the straight-fine-tune TL control pair's response evals, to
# trace the response-erosion curve across fine-tuning (the 1°/6h failure was
# erosion with training progress). For each arm × epoch in {20,40,60,80,100,120}
# (per-epoch EMA checkpoint ema_ckpt_00NN.tar), runs:
#   - +4K uniform-SST response: p0k (control) + p4k, offline constant-SST
#     perturbation inference (fme.ace.inference).
#   - future-scenario response: 46-yr trend rollout (eval-only fme.ace.train,
#     eval-trend-4deg-daily-finetune-sweep.yaml).
# 36 jobs total (6 epochs × 2 arms × [p0k + p4k + trend]).
#
# research: tasks/2026-07-08-transfer-learning-straight-finetune-controls.md
#
# Usage (run FROM this configs directory):
#   ./run-response-evals-sweep.sh                        # all 36 jobs
#   ./run-response-evals-sweep.sh reproduction-p4k-ep060 # one canary
#   ./run-response-evals-sweep.sh ep020 ep120            # substring OR filter

set -euo pipefail

# === GUARDRAILS (from research run-train.reference.sh) ======================
WANDB_IDENTITY="mcgibbon"
SCRIPT_PATH=$(git rev-parse --show-prefix)
REPO_ROOT=$(git rev-parse --show-toplevel)
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-$WANDB_IDENTITY}
if [[ "$WANDB_USERNAME" != "$WANDB_IDENTITY" ]]; then
  echo "ERROR: WANDB_USERNAME='$WANDB_USERNAME' but runs must attribute to '$WANDB_IDENTITY'." >&2
  echo "       (BEAKER_USERNAME='$BEAKER_USERNAME' would misattribute to the wandb service account.)" >&2
  exit 1
fi
if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: run this script from its own configs directory, not the repo root." >&2
  exit 1
fi
LAUNCH_FILTERS=("$@")
should_run() {
  [[ ${#LAUNCH_FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${LAUNCH_FILTERS[@]}"; do
    [[ "$1" == *"$f"* ]] && return 0
  done
  return 1
}
# === END GUARDRAILS ==========================================================

JOB_GROUP="tl-straight-finetune-response-evals-sweep"
EPOCHS=(020 040 060 080 100 120)
declare -A CKPT=(
  [reproduction]="01KXPP4WPXNVWRFZCJRG4EJKEA"
  [modern]="01KXPDRE4HDDPT4JQCFQ87YN21"
)

cd "$REPO_ROOT"

for pert in p0k p4k; do
  python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/ace-inference-era5-$pert.yaml"
done
python -m fme.ace.validate_config --config_type train "$SCRIPT_PATH/eval-trend-4deg-daily-finetune-sweep.yaml"

gantry_common=(
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")"
  --workspace ai2/ace
  --priority high
  --not-preemptible
  --cluster ai2/jupiter
  --cluster ai2/titan
  --env WANDB_USERNAME="$WANDB_USERNAME"
  --env WANDB_JOB_TYPE=inference
  --env WANDB_RUN_GROUP="$JOB_GROUP"
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa
  --dataset-secret google-credentials:/tmp/google_application_credentials.json
  --gpus 1
  --shared-memory 50GiB
  --allow-dirty
  --weka climate-default:/climate-default
  --budget ai2/atec-climate
  --system-python
  --install "pip install --no-deps ."
)

launch () {   # launch <job_name> <ckpt_dataset> <ema_epoch> <entrypoint...>
  local JOB_NAME=$1 CKPT_DATASET=$2 EMA=$3; shift 3
  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching: $JOB_NAME"
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'Response-eval epoch sweep, straight-fine-tune TL control pair' \
    --env WANDB_NAME="$JOB_NAME" \
    "${gantry_common[@]}" \
    --dataset "$CKPT_DATASET":"training_checkpoints/ema_ckpt_0$EMA.tar":/ckpt.tar \
    -- "$@"
}

for arm in reproduction modern; do
  ds="${CKPT[$arm]}"
  for ep in "${EPOCHS[@]}"; do
    launch "sweep-era5-finetune-$arm-p0k-ep$ep" "$ds" "$ep" \
      python -I -m fme.ace.inference "$SCRIPT_PATH/ace-inference-era5-p0k.yaml"
    launch "sweep-era5-finetune-$arm-p4k-ep$ep" "$ds" "$ep" \
      python -I -m fme.ace.inference "$SCRIPT_PATH/ace-inference-era5-p4k.yaml"
    launch "sweep-era5-finetune-$arm-trend46yr-ep$ep" "$ds" "$ep" \
      torchrun --nproc_per_node 1 -m fme.ace.train "$SCRIPT_PATH/eval-trend-4deg-daily-finetune-sweep.yaml"
  done
done
