#!/bin/bash

set -e

JOB_NAME_BASE="ace-aimip-evaluator-1979-2014"
JOB_GROUP="ace-aimip"
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
SEED_CHECKPOINT_IDS=("01K9B1MR70QWN90KNY7NM22K5M" \
  "01K9B1MT4QY1ZEZPPS53G2SXPK" \
  "01K9B1MVP3VS3NEABHT0W151AX" \
  "01K9B1MXD6V26S8BQH5CKY514C" \
  )
FINE_TUNED_SEED_CHECKPOINT_IDS=("01KA2F5J9768HR54369MKEHYB4"\
  "01KADMD5RAEANTP3M6GWZTXNXA" \
  "01KAC0B5ZC96GVJV46HNK9QZ9X" \
  "01KA2F5Q45122PYGW4NZBME01V" \
  )
FINE_TUNED_DOWNWEIGHTED_Q_CHECKPOINT_IDS=("01KA6NPGEQQRSZN9FF128FKJEZ"\
  "01KAF36CX46JWBZHNZYX2S7C3R" \
)
FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS=("01KAKXY0EK24K7BZK2N8SPJ5SJ"\
  "01KAVVAKANNYY096MYCGSZ7RMQ" \
  "01KAVVGKY28P5N1VA883C63EBY" \
  "01KAVVN8YPPB3P6ZSD0BGCCVX7"
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

# pre-trained
for (( i=0; i<${#SEED_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-RS$i"
    echo "Launching job $JOB_NAME checkpoint ID: ${SEED_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${SEED_CHECKPOINT_IDS[$i]}"
done

# fine-tuned
for (( i=0; i<${#FINE_TUNED_SEED_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-RS3-pressure-level-fine-tuned-RS$i"
    echo "Launching job $JOB_NAME checkpoint ID: ${FINE_TUNED_SEED_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${FINE_TUNED_SEED_CHECKPOINT_IDS[$i]}"
done

# fine-tuned with downweighted q
for  (( i=0; i<${#FINE_TUNED_DOWNWEIGHTED_Q_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-RS3-pressure-level-fine-tuned-downweighted-q-RS$i"
    echo "Launching job for fine-tuned with downweighted q seed $i checkpoint ID: ${FINE_TUNED_DOWNWEIGHTED_Q_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${FINE_TUNED_DOWNWEIGHTED_Q_CHECKPOINT_IDS[$i]}"
done

# fine-tuned with separate decoder
for  (( i=0; i<${#FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-RS3-pressure-level-fine-tuned-separate-decoder-RS$i"
    echo "Launching job for fine-tuned with separate decoder seed $i checkpoint ID: ${FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${FINE_TUNED_SEPARATE_DECODER_CHECKPOINT_IDS[$i]}"
done

# ACE2-ERA5
for  (( i=0; i<${#ACE2_ERA5_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-ACE2-ERA5-RS$i"
    echo "Launching job for ACE2-ERA5 seed $i checkpoint ID: ${ACE2_ERA5_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${ACE2_ERA5_CHECKPOINT_IDS[$i]}"
done

# ACE2-ERA5 no CO2
for  (( i=0; i<${#ACE2_ERA5_NO_CO2_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-ACE2-ERA5-no-CO2-RS$i"
    echo "Launching job for ACE2-ERA5 no CO2 seed $i checkpoint ID: ${ACE2_ERA5_NO_CO2_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${ACE2_ERA5_NO_CO2_CHECKPOINT_IDS[$i]}"
done

# ACE2-ERA5 AIMIP FF
for  (( i=0; i<${#ACE2_ERA5_AIMIP_FF_CHECKPOINT_IDS[@]}; i++ )); do
    JOB_NAME="$JOB_NAME_BASE-ACE2-ERA5-AIMIP-FF-RS$i"
    echo "Launching job for ACE2-ERA5 AIMIP FF seed $i checkpoint ID: ${ACE2_ERA5_AIMIP_FF_CHECKPOINT_IDS[$i]}"
    launch_job "$JOB_NAME" "${ACE2_ERA5_AIMIP_FF_CHECKPOINT_IDS[$i]}"
done
