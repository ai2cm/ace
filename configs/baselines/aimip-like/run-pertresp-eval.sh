#!/bin/bash
# Launch epoch-0 perturbation-response evaluations (PR #1310) of the v2
# ERA5-only checkpoints. Each job loads a source run's best-inference
# checkpoint via stepper_training.parameter_init, evaluates at epoch 0
# (evaluate_before_training, max_epochs 0 -> zero training), and runs the
# inline +4 K perturbation-response eval (whole-field and ocean-masked IC) plus
# all standard inference evals with the trend aggregator enabled.
#
# Usage: run from this directory after committing+pushing the branch:
#   GIT_REF=<sha> ./run-pertresp-eval.sh

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to repo root
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_USERNAME=${WANDB_USERNAME:-mcgibbon}
REPO_ROOT=$(git rev-parse --show-toplevel)
GIT_REF=${GIT_REF:-$(git rev-parse HEAD)}

cd "$REPO_ROOT"

run_eval() {
  local config_filename="$1"
  local job_name="$2"
  local ckpt_dataset="$3"          # beaker dataset id holding best_inference_ckpt.tar at root
  local N_GPUS="${4:-1}"
  local WORKSPACE="${5:-ai2/ace}"
  local PRIORITY="${6:-high}"
  local CLUSTER="${7:-ai2/jupiter ai2/titan}"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  local cluster_args=()
  for c in $CLUSTER; do cluster_args+=(--cluster "$c"); done

  python -m fme.ace.validate_config --config_type train "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Epoch-0 perturbation-response eval (PR #1310)' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    --ref "$GIT_REF" \
    "${cluster_args[@]}" \
    --env WANDB_USERNAME="$WANDB_USERNAME" \
    --env WANDB_NAME="$job_name" \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP= \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${ckpt_dataset}:/ckpt" \
    --gpus "$N_GPUS" \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node "$N_GPUS" -m fme.ace.train "$CONFIG_PATH"
}

# source run oshj5u79 (residual v2 ERA5-only), best-inference epoch 33
run_eval "eval-4deg-v2-era5-only-rs0-ep33-pertresp.yaml" \
  "eval-4deg-v2-era5-only-rs0-ep33-pertresp" \
  "01KW2CWN34VXNYW4XQ2KKG7TKV"

# source run znnaox7t (no-residual v2 ERA5-only), best-inference epoch 60
run_eval "eval-4deg-v2-era5-only-no-residual-rs0-ep60-pertresp.yaml" \
  "eval-4deg-v2-era5-only-no-residual-rs0-ep60-pertresp" \
  "01KW2CX5FGH3N6W7PRRXR76RKH"

# source run im4ecamc (no-residual no-co2 v2 ERA5-only, running), best-inference epoch 22
run_eval "eval-4deg-v2-era5-only-no-residual-no-co2-rs0-ep22-pertresp.yaml" \
  "eval-4deg-v2-era5-only-no-residual-no-co2-rs0-ep22-pertresp" \
  "01KW2CXC9P3VB392BHQBNMAHRX"
