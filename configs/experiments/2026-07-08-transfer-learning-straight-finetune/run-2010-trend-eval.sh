#!/bin/bash
#
# 2010-start trend eval of the MODERN arm's end-of-training (ep120 EMA)
# checkpoint: 46-yr-window remainder rollout (2010-01-01 -> late 2024, 5475
# daily steps), eval-only fme.ace.train with the trend aggregator
# (eval-trend-2010on-4deg-daily.yaml). One job. Tests out-of-sample projection
# accuracy on a shorter rollout; fig2b of ai2cm/reports#46.
#
# research: tasks/2026-07-08-transfer-learning-straight-finetune-controls.md
#
# Usage (run FROM this configs directory):
#   ./run-2010-trend-eval.sh

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
# === END GUARDRAILS ==========================================================

JOB_GROUP="tl-straight-finetune-response-evals-sweep"
MODERN_CKPT="01KXPDRE4HDDPT4JQCFQ87YN21"
JOB_NAME="sweep-era5-finetune-modern-trend2010on-ep120"

cd "$REPO_ROOT"

python -m fme.ace.validate_config --config_type train "$SCRIPT_PATH/eval-trend-2010on-4deg-daily.yaml"

gantry run \
  --name "$JOB_NAME" \
  --task-name "$JOB_NAME" \
  --description '2010-start trend eval, modern arm ep120 EMA (fig2b, reports#46)' \
  --env WANDB_NAME="$JOB_NAME" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority high \
  --not-preemptible \
  --cluster ai2/jupiter \
  --cluster ai2/titan \
  --env WANDB_USERNAME="$WANDB_USERNAME" \
  --env WANDB_JOB_TYPE=inference \
  --env WANDB_RUN_GROUP="$JOB_GROUP" \
  --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
  --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
  --dataset-secret google-credentials:/tmp/google_application_credentials.json \
  --gpus 1 \
  --shared-memory 50GiB \
  --allow-dirty \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --system-python \
  --install "pip install --no-deps ." \
  --dataset "$MODERN_CKPT":"training_checkpoints/ema_ckpt_0120.tar":/ckpt.tar \
  -- torchrun --nproc_per_node 1 -m fme.ace.train "$SCRIPT_PATH/eval-trend-2010on-4deg-daily.yaml"
