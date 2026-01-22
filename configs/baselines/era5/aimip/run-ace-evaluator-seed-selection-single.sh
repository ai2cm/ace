#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-trend-evaluation"
JOB_GROUP="ace-aimip-trend-evaluation"

ACE2_ERA5_CHECKPOINT_IDS=("01J4831KMJ5DZEE5RX4HGC1DB1" \
  "01J4874GRXA4XZAPNJY3K9NQNR" \
  "01J4MT10JPQ8MFA41F2AXGFYJ9" \
  "01J5NADA3E9PPAAT63X010B3HB" \
)
ACE2_ERA5_NO_CO2_CHECKPOINT_IDS=("01J46C2F86PKSWWY90NPQV31RY" \
  "01J4BX1YP9WG9NH6JXXVCGN58T" \
  )
ACE2_ERA5_AIMIP_FF_CHECKPOINT_IDS=("01K8EX7WMNYE86F6GP6RQPN8CH" \
  "01K8J3WTBSS13JV9NA8QX10CWK" \
)
ACE2_ERA5_AIMIP_RES_CHECKPOINT_IDS=("01K40YJST5J5JT2DX49AB280SP" \
  "01K6K2QYDNR4N4B2B8XM3HKQ9T" \
  "01K3VJBNQ7WHT2DH0JBPTWVWP5" \
  "01K3T6HBG2PF3DFBSWVBB7TCXN" \
)
ACE2_1_ERA5_PRETRAINED_CHECKPOINT_IDS=("01K9B1MR70QWN90KNY7NM22K5M" \
  "01K9B1MT4QY1ZEZPPS53G2SXPK" \
  "01K9B1MVP3VS3NEABHT0W151AX" \
  "01K9B1MXD6V26S8BQH5CKY514C" \
  )
ACE2_1_ERA5_FT_CHECKPOINT_IDS=("01KAKXY0EK24K7BZK2N8SPJ5SJ"\
  "01KAVVAKANNYY096MYCGSZ7RMQ" \
  "01KAVVGKY28P5N1VA883C63EBY" \
  "01KAVVN8YPPB3P6ZSD0BGCCVX7"
)

declare -A ALL_CHECKPOINT_ID_ENSEMBLES=(
  ["ACE2-ERA5"]=$ACE2_ERA5_CHECKPOINT_IDS \
  ["ACE2-ERA5-NO-CO2"]=$ACE2_ERA5_NO_CO2_CHECKPOINT_IDS \
  ["ACE2-ERA5-AIMIP-FF"]=$ACE2_ERA5_AIMIP_FF_CHECKPOINT_IDS \
  ["ACE2-ERA5-AIMIP-RES"]=$ACE2_ERA5_AIMIP_RES_CHECKPOINT_IDS \
  ["ACE2-1-ERA5-PRETRAINED"]=$ACE2_1_ERA5_PRETRAINED_CHECKPOINT_IDS \
  ["ACE2-1-ERA5-FT"]=$ACE2_1_ERA5_FT_CHECKPOINT_IDS \
)

IC_ENSEMBLE=(
  "1978-10-01T00:00:00" \
  "1978-10-02T00:00:00" \
  "1978-10-03T00:00:00" \
)

CONFIG_FILENAME="ace-evaluator-seed-selection-single-config.yaml"
SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
CONFIG_PATH=$SCRIPT_PATH/$CONFIG_FILENAME
BEAKER_USERNAME=bhenn1983
REPO_ROOT=$(git rev-parse --show-toplevel)

cd $REPO_ROOT  # so config path is valid no matter where we are running this script

python -m fme.ace.validate_config --config_type evaluator $CONFIG_PATH

launch_job () {

    JOB_NAME=$1
    MODEL_CHECKPOINT_DATASET=$2
    shift 2
    OVERRIDE="$@"
    echo $OVERRIDE

    cd $REPO_ROOT && gantry run \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'Run ACE2-ERA5 evaluation' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --not-preemptible \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/ceres-cirrascale \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$BEAKER_USERNAME \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset $MODEL_CHECKPOINT_DATASET:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar \
        --gpus 1 \
        --shared-memory 200GiB \
        --weka climate-default:/climate-default \
        --budget ai2/climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator $CONFIG_PATH --override $OVERRIDE

}


for CHECKPOINT_ID_ENSEMBLE_KEY in "${!ALL_CHECKPOINT_ID_ENSEMBLES[@]}"; do
  CHECKPOINT_ID_ENSEMBLE=${ALL_CHECKPOINT_ID_ENSEMBLES[$CHECKPOINT_ID_ENSEMBLE_KEY]}
  for  (( RS=0; RS<${#CHECKPOINT_ID_ENSEMBLE[@]}; RS++ )); do
    for (( IC=0; IC<${#IC_ENSEMBLE[@]}; IC++ )); do
      JOB_NAME="${JOB_NAME_BASE}-${CHECKPOINT_ID_ENSEMBLE_KEY}-RS$RS-IC$IC"
      echo "Launching job for ${CHECKPOINT_ID_ENSEMBLE_KEY} RS${RS} IC${IC}; checkpoint ID: ${CHECKPOINT_ID_ENSEMBLE[$RS]}"
      OVERRIDE="loader.start_indices.times=[${IC_ENSEMBLE[$IC]}]"
      launch_job "$JOB_NAME" "${CHECKPOINT_ID_ENSEMBLE[$RS]}" "$OVERRIDE"
    done
  done
done