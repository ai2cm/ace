#!/opt/homebrew/bin/bash

set -e

DATE="2026-02-20"
WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-som-data-only-evaluator-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

DATA_ROOT=/climate-default/2026-01-28-vertically-resolved-1deg-c96-shield-som-ensemble-fme-dataset
REFERENCE_MODEL="01KHGDAMB2BDZQS8JFF65A2YDR"

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

GCS_ROOT="gs://vcm-ml-experiments/spencerc/2026-02-20-ace-equilibrium-climate-inference"

for climate in "1xCO2" "2xCO2" "3xCO2" "4xCO2"; do
    for initial_condition in 1 2 3 4 5; do
        data_path="$climate-ic_000${initial_condition}.zarr"
        job_name=${DATE}-data-only-evaluator-$climate-ic$initial_condition
        experiment_dir="$GCS_ROOT/data-only-evaluator-$climate-ic$initial_condition/main"
        if { [ "$climate" = "1xCO2" ] || [ "$climate" = "3xCO2" ]; }; then
            override="\
                experiment_dir=$experiment_dir \
                loader.dataset.data_path=$DATA_ROOT \
                loader.dataset.file_pattern=$data_path \
                prediction_loader.dataset.data_path=$DATA_ROOT \
                prediction_loader.dataset.file_pattern=$data_path \
            "
        else
            override="\
                experiment_dir=$experiment_dir \
                loader.dataset.data_path=$DATA_ROOT \
                loader.dataset.file_pattern=$data_path \
                prediction_loader.dataset.data_path=$DATA_ROOT \
                prediction_loader.dataset.file_pattern=$data_path \
                data_writer.files=[] \
            "
        fi
        python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override $override
        gantry run \
            --name $job_name \
            --description 'Run equilibrium climate data-only evaluator' \
            --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
            --workspace ai2/ace \
            --priority high \
            --preemptible \
            --cluster ai2/jupiter \
            --cluster ai2/titan \
            --cluster ai2/ceres \
            --env WANDB_USERNAME=$WANDB_USERNAME \
            --env WANDB_NAME=$job_name \
            --env WANDB_JOB_TYPE=inference \
            --env WANDB_RUN_GROUP= \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset $REFERENCE_MODEL:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
            --gpus 1 \
            --shared-memory 20GiB \
            --weka climate-default:/climate-default \
            --budget ai2/climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- python -I -m fme.ace.evaluator $CONFIG_PATH --override $override
    done
done
