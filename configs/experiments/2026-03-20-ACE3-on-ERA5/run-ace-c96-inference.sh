
#!/bin/bash

set -e

JOB_NAME_BASE="ace-c96-multi-step-shield-ft-spencer-weights"
JOB_GROUP="ace-foundation-model"

EXISTING_RESULTS_DATASET="01KHJ5F1M6YKVZESPZAAVVD6G8"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

INFERENCE_CONFIG_FILENAME="ace-inference-era5-ssp245-gcs.yaml"
INFERENCE_CONFIG_PATH=$SCRIPT_PATH/$INFERENCE_CONFIG_FILENAME
python -m fme.ace.validate_config --config_type inference $INFERENCE_CONFIG_FILENAME

launch_job () {

    JOB_NAME=$1
    CONFIG_PATH=$2
    shift 2
    OVERRIDE="$@"

    cd $REPO_ROOT && gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 inference' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/ceres \
        --cluster ai2/titan \
        --cluster ai2/saturn \
        --cluster ai2/jupiter \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $EXISTING_RESULTS_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --allow-dirty \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference $CONFIG_PATH

}

JOB_NAME="${JOB_NAME_BASE}-ssp245-ACE-forcing-dataset-inference"
OVERRIDE=""
echo "Launching job: $JOB_NAME"
launch_job "$JOB_NAME" "$INFERENCE_CONFIG_PATH" "$OVERRIDE"

