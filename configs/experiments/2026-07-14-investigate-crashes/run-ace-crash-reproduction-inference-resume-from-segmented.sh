#!/opt/homebrew/bin/bash

set -e

DATE="2026-08-07"
WANDB_USERNAME=spencerc_ai2
CONFIG_FILENAME="ace-som-1000-year-inference-config-with-stratospheric-output.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME

INITIAL_CONDITION_ROOT=/climate-default/2026-01-28-vertically-resolved-1deg-c96-shield-som-ensemble-fme-dataset
INITIAL_CONDITION_TIME=2032-01-01T00:00:00

declare -A INITIAL_CONDITION_DATASETS
INITIAL_CONDITION_DATASETS=( \
    # ["1xCO2"]="${INITIAL_CONDITION_ROOT}/1xCO2-ic_0005.zarr" \
    # ["2xCO2"]="${INITIAL_CONDITION_ROOT}/2xCO2-ic_0005.zarr" \
    # ["3xCO2"]="${INITIAL_CONDITION_ROOT}/3xCO2-ic_0002.zarr" \
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

GCS_ROOT="gs://vcm-ml-experiments/spencerc/2026-08-07-crash-investigation-with-Rayleigh-damping"
SEGMENT_LENGTH=182621
CASES=( \
    "full-energy-conserving-rs0,4xCO2,0,498415,01KZCSW6MV219D2R29P7H7PH00,ai2/titan,example-0001" \
    "full-energy-conserving-rs0,4xCO2,1,37793,01KZCSW6WG3CYW6VXF3GN3X4Y9,ai2/titan,example-0002" \
    "full-energy-conserving-rs0,4xCO2,2,412145,01KZCSW6S00PBCR2QJWNM8PWFM,ai2/titan,example-0003" \
    "full-energy-conserving-rs0,4xCO2,5,1192075,01KZCSWKV062H0NYQ22KWB0DZ5,ai2/titan,example-0004" \
)

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT  # so config path is valid no matter where we are running this script

for case in "${CASES[@]}"; do
    IFS="," read model climate seed step_to_start_logging ic_dataset_id cluster example_name <<< $case
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
        initial_condition.engine=netcdf4 \
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

    job_name="${DATE}-${model}-${climate}-seed-${seed}-1000-year-equilibrium-climate-inference-rayleigh-damping-output-around-crash"
    gantry run \
        --name $job_name \
        --description 'Run inference with ACE' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority low \
        --cluster $cluster \
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
        --install "pip install --no-deps ." \
        -- /bin/bash -c "\
            python -I -m fme.ace.inference $CONFIG_PATH --override $spin_up_overrides \
            && \
            python -I -m fme.ace.inference $CONFIG_PATH --override $main_overrides \
        "
done
