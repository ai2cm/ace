#!/bin/bash
#
# Response evals for the straight-fine-tune transfer-learning control pair
# (reproduction + modern). Two axes of the transfer-learning verdict goal:
#   - +4K uniform-SST response: offline constant-SST-perturbation inference
#     (p0k / p2k / p4k) via fme.ace.inference.
#   - future-scenario response: 46-year trend-tracking rollout
#     (long_46year + long_46year_constant_co2) via an eval-only fme.ace.train
#     run (max_epochs=0, evaluate_before_training).
#
# Both axes evaluate best_inference_ckpt.tar (the rollout-selected checkpoint)
# for BOTH arms: the modern arm destabilized its autoregressive rollout late in
# fine-tuning, so best_inference_ckpt (its earlier rollout-stable epoch) is the
# principled, symmetric choice against the reproduction control.
#
# research: tasks/2026-07-08-transfer-learning-straight-finetune-controls.md,
#           investigations/2026-07-13-transfer-learning-straight-finetune-controls.md
#
# Usage (run FROM this configs directory):
#   ./run-response-evals.sh                 # all 8 jobs (6 pert + 2 trend)
#   ./run-response-evals.sh reproduction    # only the reproduction arm
#   ./run-response-evals.sh trend           # only the two trend jobs
#   ./run-response-evals.sh modern p4k      # substring OR filter

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

JOB_GROUP="tl-straight-finetune-response-evals"

# Fine-tune result datasets (contain training_checkpoints/best_inference_ckpt.tar).
CKPT_REPRODUCTION="01KXPP4WPXNVWRFZCJRG4EJKEA"   # eval beaker 01KXPCM7…, wandb 4wa56ocw
CKPT_MODERN="01KXPDRE4HDDPT4JQCFQ87YN21"         # eval beaker 01KXPCN0…, wandb aawlgbwz

cd "$REPO_ROOT"

# Validate all configs before paying for any GPU spin-up.
for pert in p0k p2k p4k; do
  python -m fme.ace.validate_config --config_type inference "$SCRIPT_PATH/ace-inference-era5-$pert.yaml"
done
python -m fme.ace.validate_config --config_type train "$SCRIPT_PATH/eval-trend-4deg-daily-finetune.yaml"

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

# --- +4K uniform-SST perturbation inference (fme.ace.inference) -------------
launch_pert () {   # launch_pert <job_name> <config> <ckpt_dataset>
  local JOB_NAME=$1 CONFIG_PATH=$2 CKPT_DATASET=$3
  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching (pert): $JOB_NAME"
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description 'Constant-SST-perturbation inference, straight-fine-tune TL control pair' \
    --env WANDB_NAME="$JOB_NAME" \
    "${gantry_common[@]}" \
    --dataset "$CKPT_DATASET":training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
    -- python -I -m fme.ace.inference "$CONFIG_PATH"
}

# --- 46-year future-scenario trend eval (eval-only fme.ace.train) -----------
launch_trend () {   # launch_trend <job_name> <ckpt_dataset>
  local JOB_NAME=$1 CKPT_DATASET=$2
  should_run "$JOB_NAME" || { echo "skip (filter): $JOB_NAME"; return 0; }
  echo "launching (trend): $JOB_NAME"
  gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description '46-year trend-tracking eval, straight-fine-tune TL control pair' \
    --env WANDB_NAME="$JOB_NAME" \
    "${gantry_common[@]}" \
    --dataset "$CKPT_DATASET":/weights \
    -- torchrun --nproc_per_node 1 -m fme.ace.train "$SCRIPT_PATH/eval-trend-4deg-daily-finetune.yaml"
}

for pert in p0k p2k p4k; do
  launch_pert "eval-era5-finetune-reproduction-$pert" "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_REPRODUCTION"
  launch_pert "eval-era5-finetune-modern-$pert"       "$SCRIPT_PATH/ace-inference-era5-$pert.yaml" "$CKPT_MODERN"
done

launch_trend "eval-era5-finetune-reproduction-trend46yr" "$CKPT_REPRODUCTION"
launch_trend "eval-era5-finetune-modern-trend46yr"       "$CKPT_MODERN"
