#!/bin/bash
# SST-perturbation inference for the v2 no-residual CO2 on/off A/B.
# Two 4-deg daily era5-only NO-RESIDUAL checkpoints (both finished ep120) x {+0,+4}K constant SST:
#   v2 no-residual, CO2-on   <- run znnaox7t, final ep120 result ds 01KVYDBVV6XXJZ4RQ51JVYJ607
#   v2 no-residual, CO2-off  <- run im4ecamc, final ep120 result ds 01KW0YE54G9NF9YWF000GVPMA8
# Supersedes the intermediate-ep65 znnaox7t panel used in report PR #68; fills the no-CO2 partner.
set -e

JOB_GROUP="ace-sst-pert-noresidual-co2ab-eac"

SCRIPT_PATH=$(git rev-parse --show-prefix)   # relative to repo root
REPO_ROOT=$(git rev-parse --show-toplevel)

P0K="$SCRIPT_PATH/ace-inference-era5-p0k.yaml"
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
        --description 'v2 no-residual CO2 on/off SST-pert A/B' \
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

# v2 no-residual, CO2-on  -- run znnaox7t, final ep120 result ds
CKPT_CO2="01KVYDBVV6XXJZ4RQ51JVYJ607:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar"
# v2 no-residual, CO2-off -- run im4ecamc, final ep120 result ds
CKPT_NOCO2="01KW0YE54G9NF9YWF000GVPMA8:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar"

for spec in \
    "v2-noresidual-co2-ep120-sstpert:$CKPT_CO2" \
    "v2-noresidual-noco2-ep120-sstpert:$CKPT_NOCO2" ; do
    BASE="${spec%%:*}"
    CKPT="${spec#*:}"
    echo "=== launching $BASE ==="
    launch_job "${BASE}-p0k" "$P0K" "$CKPT"
    launch_job "${BASE}-p4k" "$P4K" "$CKPT"
done
