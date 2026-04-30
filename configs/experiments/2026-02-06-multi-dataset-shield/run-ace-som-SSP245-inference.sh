#!/opt/homebrew/bin/bash

set -e

DATE="2026-04-30"
WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-som-inference-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

INITIAL_CONDITION_PATH=/climate-default/2026-04-29-ACE2S-SHiELD-SSP-reference-datasets/initial-conditions/2026-04-30-SSP-SHiELD-SOM-initial-condition-dataset.zarr

FORCING_ROOT=/climate-default/2026-04-29-ACE2S-SHiELD-SSP-reference-datasets/forcing
FORCING_PATH=2026-04-29-SSP245-CO2-with-repeating-SHiELD-SOM-forcing-1997-2099.zarr

N_FORWARD_STEPS=150476

declare -A MODELS=( \
    [published-baseline-rs3]="01J4BR6J5AW32ZDQ77VZ60P4KT" \
    # [no-random-co2-rs0]="01KHGDAMB2BDZQS8JFF65A2YDR" \
    # [no-random-co2-rs1]="01KH4SDCYN1NF2RP2JXZS0WZ1Y" \
    # [no-random-co2-energy-conserving-rs0]="01KHGDA8TVGP9JKWVJ1N0SMHCN" \
    # [no-random-co2-energy-conserving-rs1]="01KH4SDT1Q5246GZ307W8AW4M3" \
    # [full-rs0]="01KHKJ02SQM8S8T4B6030F94CV" \
    # [full-rs1]="01KHJ5EQ04XTFG46QCKX3TTAHF" \
    [full-energy-conserving-rs0]="01KHJ5F1M6YKVZESPZAAVVD6G8" \
    # [full-energy-conserving-rs1]="01KHCXABVNA3TJW0ZT5F4YDDQT" \
)

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

for model in "${!MODELS[@]}"; do
    dataset_id="${MODELS[$model]}"
    job_name=${DATE}-$model-SSP245-monthly-outputs-inference
    overrides="\
        forcing_loader.dataset.data_path=$FORCING_ROOT \
        forcing_loader.dataset.engine=zarr \
        forcing_loader.dataset.file_pattern=$FORCING_PATH \
        initial_condition.path=$INITIAL_CONDITION_PATH \
        initial_condition.engine=zarr \
        initial_condition.start_indices.times=[1997-01-01T00:00:00] \
        n_forward_steps=$N_FORWARD_STEPS \
        data_writer.files=[] \
        data_writer.save_monthly_files=true \
    "

    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $overrides
    gantry run \
        --name $job_name \
        --description 'Run inference with ACE' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/climate-titan \
        --priority urgent \
        --not-preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $dataset_id:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH --override $overrides
done
