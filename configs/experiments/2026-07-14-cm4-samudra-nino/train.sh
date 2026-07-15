#!/bin/bash

set -e

# Override CONFIG_FILENAME to run the no-nino control, e.g.:
#   CONFIG_FILENAME=train-cm4-1pct-samudra-CONTROL-no-nino.yaml JOB_NAME=cm4-1pct-control ./train.sh
CONFIG_FILENAME="${CONFIG_FILENAME:-train-cm4-1pct-samudra-nino-from-scratch.yaml}"
JOB_NAME="${JOB_NAME:-cm4-1pct-samudra-nino-train}"
JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino}"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=1

# Stats dataset that INCLUDES nino34_lead_* in ocean/centering.nc and
# ocean/scaling-full-field.nc (mounted at /ocean_stats so the config's
# /ocean_stats/ocean/*.nc paths resolve). Set this to the beaker dataset you
# uploaded (coupled-stats + nino leads).
STATS_DATA="troya/2026-07-14-cm4-1pctCO2-140yr-coupled-stats-with-nino-leads"

# NOTE: the config reads data from /climate-default (weka). Ensure these zarrs
# are staged there (see scripts/data_process/gcs_to_weka.sh):
#   2025-10-21-cm4-1pctCO2-140yr-no-smoothing-coupled-ocean.zarr
#   2025-10-16-cm4-1pctCO2-140yr-ocean-no-smoothing.zarr
#   2026-07-14-cm4-1pctCO2-140yr-ocean-no-smoothing-nino-leads.zarr

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type train $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "Ocean-only Samudra CM4 1pctCO2 from scratch with Nino3.4 readout head" \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --preemptible \
    --cluster ai2/titan \
    --weka climate-default:/climate-default \
    --env WANDB_USERNAME=$BEAKER_USERNAME \
    --env WANDB_NAME=$JOB_NAME \
    --env WANDB_JOB_TYPE=training \
    --env WANDB_RUN_GROUP=$JOB_GROUP \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --dataset $STATS_DATA:/ocean_stats \
    --gpus $N_GPUS \
    --shared-memory 400GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.ace.train $CONFIG_PATH
