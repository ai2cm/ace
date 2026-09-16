#!/bin/bash
# Launch the checkpoint-ensemble stability probes: residfixbest (residual)
# blended with the corrected full-field pretrain at several weights, 20-yr
# truth-forced ocean-only rollouts on the standard probe ICs.
set -euo pipefail

ARMS="${ARMS:-ff00 ff10 ff20 ff50 ff100}"
PRIORITY="${PRIORITY:-urgent}"
# residfixbest: samudra-enso-w1-residfix results, best-validation checkpoint
# (the run has no best_inference_ckpt: every 20-yr inference NaN'd).
RESID_DS="${RESID_DS:-01M1FKAM7N8MKN7ZTANMB2N38Y}"
# pretrain0: cm4-samudra-1pct-ocean-train-using-ufs-var-subset-ohc-hdfs-correctors
FF_DS="${FF_DS:-01KW2BQ83EGZ90WZ74CZ4TJATN}"

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#"$REPO_ROOT"/}
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
cd "$REPO_ROOT"

for arm in $ARMS; do
  config="${SCRIPT_PATH}/probe-${arm}.yaml"
  job="samudra-enso-ckptens-${arm}"
  gantry run \
    --name "$job" --task-name "$job" \
    --description "Checkpoint-ensemble probe ${arm}: residfixbest + full-field blend, 20-yr ocean-only rollout" \
    --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
    --workspace ai2/ace --priority "$PRIORITY" --preemptible \
    --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME="$BEAKER_USERNAME" --env WANDB_NAME="$job" \
    --env WANDB_JOB_TYPE=evaluation --env WANDB_RUN_GROUP=samudra-enso-ckptens \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset "${RESID_DS}:training_checkpoints/best_ckpt.tar:/ckpt_resid.tar" \
    --dataset "${FF_DS}:training_checkpoints/best_inference_ckpt.tar:/ckpt_ff.tar" \
    --gpus 1 --shared-memory 400GiB --budget ai2/atec-climate \
    --allow-dirty --system-python --install "pip install --no-deps ." \
    -- bash -c "python '${SCRIPT_PATH}/make_probe_configs.py' --arms ${arm} && python -I -m fme.ace.evaluator '$config'"
done
