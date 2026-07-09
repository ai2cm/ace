#!/bin/bash
# One launcher for all ACE2S land-forcing short-lead evaluation runs.
# Each run is a single-GPU `fme.ace.evaluator` job that rolls the model 40 steps (~10 days) from
# ~300 ICs, computes near-surface skill vs the target, and writes paired prediction+target fields
# to a GCS zarr. Two dataset configs (era5.yaml, cm4.yaml) are reused across checkpoints; the
# checkpoint is mounted at /ckpt.tar and the per-run experiment_dir is set via --override.
#
# Usage:
#   ./run-inference.sh                 # submit every run
#   ./run-inference.sh era5            # only runs whose config/job name matches this substring
#   ./run-inference.sh snow            # e.g. just the two snow treatments
# Recommended: run one pilot (small n_initial_conditions / n_forward_steps to a scratch path)
# first, confirm the GCS zarr has the derived + component vars, then submit the rest.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"          # this dir relative to the repo root
WANDB_USERNAME="bhenn1983"
JOB_GROUP="ace2s-land-forcing-eval"
GCS_PREFIX="gs://vcm-ml-intermediate/2026-07-09-ace2s-land-forcing-inference"
SELECT="${1:-}"                                   # optional substring filter; empty => all

# --- Checkpoint beaker dataset IDs (best_inference_ckpt.tar) -------------------------------------
# Controls: the deployed ACE2S checkpoints (no extra land forcing), reused as baseline.
ERA5_CONTROL=01KSVC6YS7C18SGYV4VPZYZ232
CM4_CONTROL_RS0=01KTYXNSJX90Y5E2CQ6SV8K37D
# CM4_CONTROL_RS1=01KTWGH2VEZ4DNXXF1H5FTJK1S   # optional second control seed

# Treatments: the *1-step pretrain* land-forcing checkpoints (NOT the multi-step finetuned ones —
# long rollouts are meteorologically confounded by the leaked land forcing). Fill in the four
# 1-step-pretrain beaker dataset IDs before submitting the treatment runs.
# Most-recent *committed* result dataset per pretrain experiment (best_ckpt.tar lives under
# training_checkpoints/). Result-dataset IDs are ULIDs, so they sort in retry order.
ERA5_SNOW=01KX47AYFXZR5GBP5236ZQ2H8G   # newest retry (captures the final epoch); confirm it has
                                       # committed before launching (prior committed: 01KX1S6C2G5E4856CSV70QN544).
ERA5_SOIL=01KWZDSCZ69NPBR39H1JS946RF
CM4_SNOW=01KX017K0Z48ZW91769NF7MQXH
CM4_SOIL=01KX1RVPA4YQYZK9NK76ZEPWMJ

# 4th arg = checkpoint file within the beaker dataset. Treatments use best_ckpt.tar (lowest
# *validation* loss — the short-horizon criterion, and what multi-step finetuning warm-started
# from); the pretrain jobs' best_inference_ckpt.tar is selected on a 7300-step rollout, i.e. the
# confounded long-rollout regime, so it is not used here. Controls default to best_inference_ckpt
# (their as-deployed checkpoint).
submit() {
    local config="$1" ckpt="$2" job="$3" ckpt_file="${4:-training_checkpoints/best_ckpt.tar}"
    if [ -n "$SELECT" ] && [[ "$config" != *"$SELECT"* ]] && [[ "$job" != *"$SELECT"* ]]; then
        return 0
    fi
    if [ -z "$ckpt" ]; then
        echo "SKIP $job: checkpoint ID not set" >&2
        return 0
    fi
    local config_path="${CONFIG_DIR}/${config}"
    python -m fme.ace.validate_config --config_type evaluator "$config_path"
    gantry run \
        --name "$job" \
        --task-name "$job" \
        --description "ACE2S land-forcing eval: ${job}" \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/ace \
        --priority high \
        --preemptible \
        --cluster ai2/titan \
        --env WANDB_USERNAME="$WANDB_USERNAME" \
        --env WANDB_NAME="$job" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$ckpt:${ckpt_file}:/ckpt.tar" \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.evaluator "$config_path" \
             --override "experiment_dir=${GCS_PREFIX}/${job}"
}

cd "$REPO_ROOT"  # so the config paths resolve regardless of where this is run from

# config       checkpoint         job name                 [checkpoint file]
# Treatments (1-step pretrain) use the default best_ckpt.tar; controls use their as-deployed
# best_inference_ckpt.tar.
submit era5.yaml "$ERA5_CONTROL"    lf-eval-era5-control     training_checkpoints/best_inference_ckpt.tar
submit era5.yaml "$ERA5_SNOW"       lf-eval-era5-snow
submit era5.yaml "$ERA5_SOIL"       lf-eval-era5-soil
submit cm4.yaml  "$CM4_CONTROL_RS0" lf-eval-cm4-control-rs0  training_checkpoints/best_inference_ckpt.tar
submit cm4.yaml  "$CM4_SNOW"        lf-eval-cm4-snow
submit cm4.yaml  "$CM4_SOIL"        lf-eval-cm4-soil
