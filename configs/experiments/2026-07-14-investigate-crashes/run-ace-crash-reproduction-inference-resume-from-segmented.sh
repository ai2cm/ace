#!/opt/homebrew/bin/bash

set -e

DATE="2026-07-22"
WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-som-1000-year-inference-config-with-stratospheric-output.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

GCS_ROOT="gs://vcm-ml-experiments/spencerc/2026-07-14-crash-investigation/example-0002"

INITIAL_CONDITION_ROOT=/climate-default/2026-01-28-vertically-resolved-1deg-c96-shield-som-ensemble-fme-dataset
INITIAL_CONDITION_TIME=2032-01-01T00:00:00

declare -A INITIAL_CONDITION_DATASETS
INITIAL_CONDITION_DATASETS=( \
    ["1xCO2"]="${INITIAL_CONDITION_ROOT}/1xCO2-ic_0005.zarr" \
    ["2xCO2"]="${INITIAL_CONDITION_ROOT}/2xCO2-ic_0005.zarr" \
    ["3xCO2"]="${INITIAL_CONDITION_ROOT}/3xCO2-ic_0002.zarr" \
    ["4xCO2"]="${INITIAL_CONDITION_ROOT}/4xCO2-ic_0005.zarr" \
)

declare -A CO2_CONCENTRATIONS
CO2_CONCENTRATIONS=( \
    ["1xCO2"]=0.00036343 \
    ["2xCO2"]=0.00072686 \
    ["3xCO2"]=0.00109029 \
    ["4xCO2"]=0.00145372 \
)

declare -A MODELS=( \
    [full-energy-conserving-rs0]="01KHJ5F1M6YKVZESPZAAVVD6G8" \
)

GCS_ROOT="gs://vcm-ml-experiments/spencerc/2026-07-14-crash-investigation"
SEGMENT_LENGTH=182621
CASES=( \
    "full-energy-conserving-rs0,2xCO2,4,176821,01KY2Y1KHDG38PB98DKDN0ZDHQ,example-0003" \
    # "full-energy-conserving-rs0,3xCO2,3,398052,01KY02N19VKSX3NHQEA0W9AK5M,example-0004" \
    # "full-energy-conserving-rs0,3xCO2,4,267230,01KY2Y1K2YZP8JHXP09Y63F5T1,example-0005" \
    # "full-energy-conserving-rs0,4xCO2,4,537921,01KY02N8ZMGTVRP5EWVGTEZJZ5,example-0006" \
)

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

for case in "${CASES[@]}"; do
    IFS="," read model climate seed step_to_start_logging ic_dataset_id example_name <<< $case
    co2_concentration=${CO2_CONCENTRATIONS[$climate]}
    dataset_id=${MODELS[$model]}
    initial_condition_segment=$(printf "%04d" $((step_to_start_logging / SEGMENT_LENGTH)))
    spin_up_steps=$((step_to_start_logging % SEGMENT_LENGTH))
    initial_condition_path="/spun_up_initial_condition.nc"
    gcs_root="${GCS_ROOT}/${example_name}"

    spin_up_overrides="\
        forcing_loader.dataset.overwrite.constant.global_mean_co2=$co2_concentration \
        initial_condition.path=$initial_condition_path \
        initial_condition.start_indices.list=[0] \
        n_forward_steps=$spin_up_steps \
        experiment_dir=$gcs_root/spin-up \
        logging.log_to_wandb=false \
        data_writer.files=[] \
        seed=$seed \
    "
    main_overrides="\
        forcing_loader.dataset.overwrite.constant.global_mean_co2=$co2_concentration \
        initial_condition.path=$gcs_root/spin-up/restart.nc \
        initial_condition.start_indices.list=[0] \
        initial_condition.engine=netcdf4 \
        n_forward_steps=1460 \
        experiment_dir=$gcs_root/main \
        seed=$seed \
    "

    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $spin_up_overrides
    python -m fme.ace.validate_config --config_type inference $CONFIG_PATH --override $main_overrides

    job_name="${DATE}-${model}-${climate}-seed-${seed}-1000-year-equilibrium-climate-inference-output-around-crash"
    gantry run \
        --name $job_name \
        --description 'Run inference with ACE' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --cluster ai2/titan \
        --env WANDB_USERNAME=$WANDB_USERNAME \
        --env WANDB_NAME=$job_name \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP= \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $dataset_id:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --dataset $ic_dataset_id:segment_${initial_condition_segment}/initial_condition.nc:/spun_up_initial_condition.nc \
        --gpus 1 \
        --shared-memory 20GiB \
        --weka climate-default:/climate-default \
        --system-python \
        --min-runtime 8h \
        --install "pip install --no-deps ." \
        -- /bin/bash -c "\
            python -I -m fme.ace.inference $CONFIG_PATH --override $spin_up_overrides \
            && \
            python -I -m fme.ace.inference $CONFIG_PATH --override $main_overrides \
        "
done
