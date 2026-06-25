#!/bin/bash
# SST-perturbation inference for the EAC v2-baseline panels.
# Two 4-deg daily era5-only checkpoints x {+0,+2,+4}K constant SST.
#   v2 baseline (residual+clip)     <- run j8r0z322, finalized ckpt ds
#   v2 baseline (no residual)       <- run znnaox7t, intermediate (epoch ~65) ckpt snapshot
set -e

JOB_GROUP="ace-sst-pert-v2-baseline-eac"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

SCRIPT_PATH=$(git rev-parse --show-prefix)   # relative to repo root
REPO_ROOT=$(git rev-parse --show-toplevel)

P0K="$SCRIPT_PATH/ace-inference-era5-p0k.yaml"
P2K="$SCRIPT_PATH/ace-inference-era5-p2k.yaml"
P4K="$SCRIPT_PATH/ace-inference-era5-p4k.yaml"

cd "$REPO_ROOT"
python -m fme.ace.validate_config --config_type inference "$P0K"
python -m fme.ace.validate_config --config_type inference "$P4K"

# launch_job <job_name> <config_path> <ckpt_mount_spec>
launch_job () {
    JOB_NAME=$1
    CONFIG_PATH=$2
    CKPT_MOUNT=$3
    cd "$REPO_ROOT" && gantry run \
        --yes \
        --no-logs \
        --name "$JOB_NAME" \
        --task-name "$JOB_NAME" \
        --description 'EAC v2-baseline SST-pert inference' \
        --beaker-image "$(cat "$REPO_ROOT"/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --cluster ai2/titan \
        --cluster ai2/jupiter \
        --env WANDB_USERNAME=mcgibbon \
        --env WANDB_NAME="$JOB_NAME" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$CKPT_MOUNT" \
        --gpus 1 \
        --shared-memory 50GiB \
        --allow-dirty \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference "$CONFIG_PATH"
}

# v2 baseline (residual+clip) -- run j8r0z322, finalized result ds
CKPT_RES="01KVEBK3DP4JEXGEAJK1CAAXFY:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar"
# v2 baseline (no residual) -- run znnaox7t, intermediate epoch-65 snapshot ds
CKPT_NORES="01KVZKDXKGQK2QGS2RR9P9HYSP:best_inference_ckpt.tar:/ckpt.tar"

for spec in \
    "v2-baseline-residual-sstpert:$CKPT_RES" \
    "v2-baseline-noresidual-ep65-sstpert:$CKPT_NORES" ; do
    BASE="${spec%%:*}"
    CKPT="${spec#*:}"
    echo "=== launching $BASE ==="
    launch_job "${BASE}-p0k" "$P0K" "$CKPT"
    launch_job "${BASE}-p2k" "$P2K" "$CKPT"
    launch_job "${BASE}-p4k" "$P4K" "$CKPT"
done
