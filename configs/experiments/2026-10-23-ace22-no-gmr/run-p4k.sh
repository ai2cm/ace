#!/bin/bash

set -e
export GRPC_VERBOSITY=ERROR

# Control and +4 K SST inference of the selected stage-2 checkpoint (best_inference_ckpt.tar),
# as trained and with latent global-mean clipping disabled. Scored exactly as ACE2.2-ERA5's
# four seeds were (ace22-seed-perturbation-rs0..3{,-p4k}, from
# ../2026-08-28-ace22-variants/run-seed-perturbation.sh on its branch): the same configs
# (inference-{control,p4k}.yaml; only the file labels' model token differs), code, image,
# initial condition and inference seed. The rollouts run 1978-10 to 2024-12 like those runs;
# they come after checkpoint selection and feed no selection.
#
#   bash run-p4k.sh
#
# The clip-off jobs need dataset ace22-no-gmr-stage2-final-clipoff
# (bash snapshot-clip-off.sh final best_inference_ckpt.tar). Safe to rerun: existing
# experiments are skipped.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE2="ace22-no-gmr-stage2-rs0"
CKPT_FILE="best_inference_ckpt.tar"
ACE_GIT_REF="394e41b5e07d1a3606d51faea1bd09a6fbddb492"
IMAGE="oliverwm/fme-deps-only-c82095065"
OUTPUT_ROOT="/climate-default/2026-10-27-ace22-no-gmr-perturbation"
IC_PATH="/climate-default/2026-08-24-aimip-evaluation/aimip-evaluation-ics/1978-09-30_IC0.nc"
CONFIG_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$SCRIPT_DIR" -c core.fileMode=false status --porcelain -- .)" ]]; then
    CONFIG_COMMIT="$CONFIG_COMMIT-dirty"
fi
BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
WANDB_IDENTITY="bhenn1983"  # differs from BEAKER_USERNAME; do not derive one from the other

STATUS=$(beaker experiment get "$BEAKER_USERNAME/$STAGE2" --format json \
    | python3 -c "import json, sys; j = json.load(sys.stdin)[0]['jobs'][-1]; print(j['status'].get('exitCode', ''), j['result']['beaker'])")
read -r EXIT_CODE RESULT <<< "$STATUS"
if [[ "$EXIT_CODE" != 0 ]]; then
    echo "ERROR: $STAGE2 has not exited 0" >&2
    exit 1
fi
CLIPOFF="$BEAKER_USERNAME/ace22-no-gmr-stage2-final-clipoff"
CLIPOFF_ID=$(beaker dataset get "$CLIPOFF" --format json 2>/dev/null | jq -r '.[0].id' || true)

for CLIP in on off; do
    if [[ $CLIP == on ]]; then
        SUFFIX=""
        MOUNT="$RESULT:training_checkpoints/$CKPT_FILE:/ckpt.tar"
    else
        SUFFIX="-clipoff"
        if [[ -z "$CLIPOFF_ID" ]]; then
            echo "wait clip-off jobs: no dataset $CLIPOFF"
            continue
        fi
        MOUNT="$CLIPOFF_ID:$CKPT_FILE:/ckpt.tar"
    fi
    for EXP in control p4k; do
        JOB_NAME="ace22-no-gmr-perturbation-$EXP$SUFFIX"
        if beaker experiment get "$BEAKER_USERNAME/$JOB_NAME" >/dev/null 2>&1; then
            echo "skip $JOB_NAME: experiment exists"
            continue
        fi
        CONFIG_B64=$(base64 < "$SCRIPT_DIR/inference-$EXP.yaml" | tr -d '\n')
        # seed=1 is the inference noise draw ACE2.2's seed runs used.
        OVERRIDE="initial_condition.path=$IC_PATH experiment_dir=$OUTPUT_ROOT/$JOB_NAME seed=1"
        echo "launch $JOB_NAME"
        gantry run \
            --remote https://github.com/ai2cm/ace \
            --ref "$ACE_GIT_REF" \
            --name "$JOB_NAME" \
            --task-name "$JOB_NAME" \
            --description "ACE2.2-ERA5 without GMR: $EXP inference of stage-2 $CKPT_FILE, clip $CLIP" \
            --beaker-image "$IMAGE" \
            --workspace ai2/ace \
            --priority high \
            --min-runtime 8h \
            --cluster ai2/titan \
            --cluster ai2/jupiter-cirrascale-2 \
            --env WANDB_USERNAME=$WANDB_IDENTITY \
            --env WANDB_NAME="$JOB_NAME" \
            --env WANDB_JOB_TYPE=inference \
            --env WANDB_RUN_GROUP=ace22-no-gmr \
            --env EXPERIMENT_CONFIG_COMMIT="$CONFIG_COMMIT" \
            --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
            --env-secret WANDB_API_KEY=wandb-api-key-${BEAKER_USERNAME} \
            --dataset-secret google-credentials:/tmp/google_application_credentials.json \
            --dataset "$MOUNT" \
            --gpus 1 \
            --shared-memory 50GiB \
            --weka climate-default:/climate-default \
            --budget ai2/atec-climate \
            --system-python \
            --install "pip install --no-deps ." \
            -- bash -c "echo '${CONFIG_B64}' | base64 -d > /tmp/config.yaml && python -I -m fme.ace.inference /tmp/config.yaml --override ${OVERRIDE}"
    done
done
