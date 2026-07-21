#!/bin/bash
#
# Coupled (SamudrACE) fine-tuning of the atmosphere on CM4 1pctCO2, with the
# ocean (Samudra + Nino3.4 readout head) frozen.
#
# Atmosphere weights: jamesd's uncoupled condSFNO fine-tune
#   (experiment 01KHZG19J8SFKZ3PDYCCEZNH1X -> result dataset below).
# Ocean weights: our nino-head Samudra training run.

set -e

CONFIG_FILENAME="${CONFIG_FILENAME:-coupled-finetune-atmos.yaml}"
JOB_NAME="${JOB_NAME:-cm4-1pct-coupled-ft-atmos-nino-ocean}"
JOB_GROUP="${JOB_GROUP:-cm4-1pct-samudra-nino}"

ATMOS_RESULTS_DATASET="01KJ70WK2NH4T2T4AVAAPYFSHA"  # aceS-cm4_1pctCO2_0256to0350-condSFNO-rs0-train
OCEAN_RESULTS_DATASET="01KXKZ85HTDSGGXWD2DPW2QRFW"  # cm4-1pct-samudra-nino-train
ATMOS_CKPT="best_inference_ckpt"
OCEAN_CKPT="best_inference_ckpt"

# Atmosphere stats: same dataset jamesd's run used (coupled_atmosphere subdir).
ATMOS_STATS_DATA="01KHGYVHSX504ZBHJC223S63F0"
# Ocean stats INCLUDING nino34_lead_* (config uses /ocean_stats/ocean/*.nc).
OCEAN_STATS_DATA="troya/2026-07-14-cm4-1pctCO2-140yr-coupled-stats-with-nino-leads"

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH="${SCRIPT_PATH}${CONFIG_FILENAME}"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
REPO_ROOT=$(git rev-parse --show-toplevel)
N_GPUS=4

# NOTE: the config reads data from /climate-default (weka). Ensure these zarrs
# are staged there (see scripts/data_process/gcs_to_weka.sh):
#   2025-10-21-cm4-1pctCO2-140yr-no-smoothing-coupled-ocean.zarr
#   2025-10-16-cm4-1pctCO2-140yr-ocean-no-smoothing.zarr
#   2026-07-14-cm4-1pctCO2-140yr-ocean-no-smoothing-nino-leads.zarr
#   2025-10-21-cm4-1pctCO2-140yr-no-smoothing-coupled-sea_ice.zarr
#   2025-06-18-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr.zarr

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.coupled.validate_config --config_type train $CONFIG_PATH

gantry run \
    --name $JOB_NAME \
    --task-name $JOB_NAME \
    --description "Coupled FT: condSFNO atmosphere + frozen nino-head Samudra ocean (CM4 1pctCO2)" \
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
    --dataset "$ATMOS_STATS_DATA:coupled_atmosphere:/atmos_stats" \
    --dataset "$OCEAN_STATS_DATA:/ocean_stats" \
    --dataset "$ATMOS_RESULTS_DATASET:training_checkpoints/${ATMOS_CKPT}.tar:/atmos_ckpt.tar" \
    --dataset "$OCEAN_RESULTS_DATASET:training_checkpoints/${OCEAN_CKPT}.tar:/ocean_ckpt.tar" \
    --gpus $N_GPUS \
    --shared-memory 600GiB \
    --budget ai2/atec-climate \
    --allow-dirty \
    --system-python \
    --install "pip install --no-deps ." \
    -- torchrun --nproc_per_node $N_GPUS -m fme.coupled.train $CONFIG_PATH
