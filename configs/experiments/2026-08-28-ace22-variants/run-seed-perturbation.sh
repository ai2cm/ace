#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Forced response across the ACE2.2 seed ensemble: control and +4K SST inference on all
# four seeds' stage-2 checkpoints, one realization each. 8 jobs, ~1.75 h on 1 GPU apiece.
#
# The question is whether the ~50% shortfall in ACE2.2's response to uniform SST warming
# (2.0 K against GFDL-CM4's 4.6 K) varies with the seed. Only submitted models get
# perturbation runs, so this rests on two points today -- seed 0's and seed 3's production
# sweeps -- and those differ by 3% while their historical trend shares differ by 27 points.
# Four seeds cannot explain the shortfall, but they can say whether seed selection is a
# lever on it at all, and they supply the across-seed test of whether the lowest-bias seed
# has the weakest response (hypothesis 5 in the findings report).
#
# Stage 2, not stage 3: near-surface temperature is prognostic in ACE2.2, so tas exists
# without the pressure-level decoder. The configs are the stage-1 probe's, which already
# drop the 12 `gr` ta/hus/ua/va entries the secondary decoder would supply; zg (h500) and
# TMP850 are core outputs and remain. Stages 1 and 2 share their output set, so those
# configs are valid here unchanged apart from the source_id token.
#
# ONE ace ref for all eight jobs. The four checkpoints were trained at different commits,
# but fme/ is byte-identical across them, so pinning one ref keeps the comparison free of
# any inference-code difference.
#
# Every job writes the same filenames (all labelled r1i1p1f1) into its own directory: the
# seeds are different models, not realizations of one, so they are kept apart rather than
# stacked on a realization axis.

JOB_NAME_BASE="ace22-seed-perturbation"
JOB_GROUP="ace22-seed-perturbation"
SEEDS="${SEEDS:-0 1 2 3}"
EXPERIMENTS="${EXPERIMENTS:-control p4k}"
MIN_RUNTIME="${MIN_RUNTIME:-8h}"

# Each seed's stage-2 result dataset, from the 2026-08-12 experiment's run-train.sh.
# Selected checkpoints (epochs 8, 4, 20, 24) -- the model each seed's protocol would ship,
# and the basis seed 0's and seed 3's production numbers already sit on. Set
# CHECKPOINT_FILE=ema_ckpt_0040.tar to rerun on matched training length instead.
declare -A STAGE2_DATASET=(
  [0]=01M0RFP2DKAGABV89KRPMXX5C3
  [1]=01M1NGC06ZYNV8CNX62WE7JQ4P
  [2]=01M1NGNEN34KX4BVKWPTNFA2YG
  [3]=01M1NHPDTXGJGJBBCP24H42DRK
)
CHECKPOINT_FILE="${CHECKPOINT_FILE:-best_inference_ckpt.tar}"

ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
OUTPUT_ROOT="${OUTPUT_ROOT:-/climate-default/2026-09-08-ace22-seed-perturbation}"
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
CFG_DIR=$SCRIPT_DIR/seed-perturbation

launch_job () {
    local JOB_NAME=$1 TEMPLATE_CONFIG=$2 DATASET=$3 OVERRIDE=$4
    local CONFIG_B64
    CONFIG_B64=$(base64 < "$TEMPLATE_CONFIG" | tr -d '\n')

    gantry run \
        --remote https://github.com/ai2cm/ace \
        --ref $ACE_GIT_REF \
        --name $JOB_NAME \
        --task-name $JOB_NAME \
        --description 'ACE2.2 seed-ensemble forced-response probe (stage-2 checkpoints)' \
        --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
        --workspace ai2/ace \
        --priority high \
        --min-runtime "$MIN_RUNTIME" \
        --cluster ai2/titan-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --env WANDB_USERNAME=$WANDB_IDENTITY \
        --env WANDB_NAME=$JOB_NAME \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP=$JOB_GROUP \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset ${DATASET}:training_checkpoints/${CHECKPOINT_FILE}:/ckpt.tar \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/seed-perturbation-config.yaml && python -I -m fme.ace.inference /tmp/seed-perturbation-config.yaml --override ${OVERRIDE}"
}

for SEED in $SEEDS; do
    DATASET=${STAGE2_DATASET[$SEED]:-}
    if [ -z "$DATASET" ]; then
        echo "ERROR: no stage-2 dataset recorded for seed '$SEED'." >&2
        exit 1
    fi
    for EXPERIMENT in $EXPERIMENTS; do
        case $EXPERIMENT in
            control) CFG=$CFG_DIR/ace-seed-perturbation-inference-config.yaml;     SUFFIX="" ;;
            p4k)     CFG=$CFG_DIR/ace-seed-perturbation-inference-p4k-config.yaml; SUFFIX="-p4k" ;;
            *) echo "ERROR: EXPERIMENT must be 'control' or 'p4k', got '$EXPERIMENT'." >&2; exit 1 ;;
        esac
        JOB_NAME="${JOB_NAME_BASE}-rs${SEED}${SUFFIX}"
        # seed=1 fixes the inference noise draw, so the four models are compared on one
        # draw rather than differing by both training seed and rollout noise.
        OVERRIDE="initial_condition.path=${IC_PATH} experiment_dir=${OUTPUT_ROOT}/${JOB_NAME} seed=1"
        echo "Launching $JOB_NAME  (stage-2 dataset $DATASET, $CHECKPOINT_FILE)"
        launch_job "$JOB_NAME" "$CFG" "$DATASET" "$OVERRIDE"
    done
done
