#!/bin/bash
# Launch SE Asia / Singapore 10-year single-patch downscaling: HiRO-ACE on 5 ACE2S ICs
# plus HiRO perfect prediction. Adapted from ../downscale.sh; see sea-singapore-10yr-PROVENANCE.md.
# Pass --dry-run to forward it to gantry.

set -e

SCRIPT_PATH=$(echo "$(git rev-parse --show-prefix)" | sed 's:/*$::')

# since we use a service account API key for wandb, we use the beaker username to set the wandb username
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

GANTRY=${GANTRY:-gantry}
NGPU=2

IMAGE="$(cat latest_deps_only_image.txt)"
EXISTING_RESULTS_DATASET_HIGH_SIGMA=01KRGZT4X2QCW2RFH7WN7X8BYA
EXISTING_RESULTS_DATASET_LOW_SIGMA=01KRBYGNYJ6FD7PGNF3VVHQ5V1

wandb_group=""

declare -A JOBS=(
    ["hiro-perfect-pred-inference-sea-singapore-2014-2023"]="downscale-perfect-pred-sea-singapore-10yr.yaml"
)
for IC in 0000 0001 0002 0003 0004; do
    JOBS["hiro-ace2s-inference-sea-singapore-2014-2023-ic${IC}"]="downscale-ace2s-sea-singapore-10yr-ic${IC}.yaml"
done

for JOB_NAME in "${!JOBS[@]}"; do
    CONFIG_PATH=$SCRIPT_PATH/${JOBS[$JOB_NAME]}
    $GANTRY run \
        --name $JOB_NAME \
        --description 'Run 100km to 3km single-patch downscaling over SE Asia / Singapore, 2014-2023' \
        --workspace ai2/ace \
        --priority high \
        --min-runtime 2h \
        --cluster ai2/titan \
        --beaker-image $IMAGE \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$wandb_group \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET_HIGH_SIGMA:checkpoints:/checkpoints_high_sigma  \
        --dataset $EXISTING_RESULTS_DATASET_LOW_SIGMA:checkpoints:/checkpoints_low_sigma  \
        --weka climate-default:/climate-default \
        --gpus $NGPU \
        --shared-memory 200GiB \
        --budget ai2/atec-climate \
        --no-python \
        --install "pip install --no-deps ." \
        --yes \
        "$@" \
        -- torchrun --nproc_per_node $NGPU -m fme.downscaling.inference $CONFIG_PATH
done
