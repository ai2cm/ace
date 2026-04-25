#!/opt/homebrew/bin/bash

set -e

DATE="2026-04-25"
CONFIG_FILENAME="ace-amip-data-only-evaluator-config-daily-PRATEsfc.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
 # since we use a service account API key for wandb, we use the beaker username to set the wandb username
WANDB_USERNAME=spencerc_ai2
REPO_ROOT=$(git rev-parse --show-toplevel)

CHECKPOINT_PATH=training_checkpoints/best_inference_ckpt.tar

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

REFERENCE_MODEL="01KHGDAMB2BDZQS8JFF65A2YDR"

GCS_ROOT="gs://vcm-ml-experiments/spencerc/2026-04-25-amip-inference"
TRAIN_AND_VALIDATE_EXPERIMENT_DIR="/results/train-and-validate"

# xr.date_range("1980", "2012", freq="6h", inclusive="left")
TRAIN_AND_VALIDATE_START_TIME="1980-01-01T00:00:00"
TRAIN_AND_VALIDATE_N_FORWARD_STEPS=46752

# xr.date_range("2012", "2021", freq="6h", inclusive="left")
TEST_START_TIME="2012-01-01T00:00:00"
TEST_N_FORWARD_STEPS=13152

for ensemble_id in "ic_0001"; do
    test_experiment_dir="${GCS_ROOT}/SHiELD-${ensemble_id}/test"
    job_name="${DATE}-amip-${ensemble_id}-split-data-only-evaluator"
    train_and_validate_override="\
        experiment_dir=${TRAIN_AND_VALIDATE_EXPERIMENT_DIR} \
        n_forward_steps=${TRAIN_AND_VALIDATE_N_FORWARD_STEPS} \
        loader.start_indices.times=[${TRAIN_AND_VALIDATE_START_TIME}] \
        prediction_loader.start_indices.times=[${TRAIN_AND_VALIDATE_START_TIME}] \
        loader.dataset.file_pattern=${ensemble_id}.zarr \
        prediction_loader.dataset.file_pattern=${ensemble_id}.zarr \
        data_writer.files=[] \
    "
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override $train_and_validate_override
    test_override="\
        experiment_dir=${test_experiment_dir} \
        n_forward_steps=${TEST_N_FORWARD_STEPS} \
        loader.start_indices.times=[${TEST_START_TIME}] \
        prediction_loader.start_indices.times=[${TEST_START_TIME}] \
        loader.dataset.file_pattern=${ensemble_id}.zarr \
        prediction_loader.dataset.file_pattern=${ensemble_id}.zarr \
    "
    python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH --override $test_override

    gantry run \
        --name $job_name \
        --description 'Run ACE AMIP data-only evaluator' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/jupiter \
        --cluster ai2/ceres \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $REFERENCE_MODEL:$CHECKPOINT_PATH:/ckpt.tar \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- /bin/bash -c "\
        python -I -m fme.ace.evaluator $CONFIG_PATH --override $train_and_validate_override \
        && \
        python -I -m fme.ace.evaluator $CONFIG_PATH --override $test_override \
        "
done
