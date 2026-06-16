#!/bin/bash
# One launcher for all ACE2S land-feedback inference runs (experiments E1-E7).
# Common settings (clusters, image, budget, secrets) live once in submit(); each run is one call at
# the bottom. Edit common behavior in one place.
#
# Usage:
#   ./run-inference.sh                 # submit every run
#   ./run-inference.sh pilot-era5      # submit only runs whose config/job matches this substring
#   ./run-inference.sh frameworkB-cm4  # e.g. just the two CM4 Framework-B seeds
# Recommended: run the pilot first, confirm the GCS zarr ingests, then submit the rest.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"          # this dir relative to the repo root
WANDB_USERNAME="bhenn1983"
JOB_GROUP="ace2s-land-feedback"
SELECT="${1:-}"                                   # optional substring filter; empty => all

# checkpoint beaker dataset IDs (best_inference_ckpt.tar)
ERA5=01KSVC6YS7C18SGYV4VPZYZ232
CM4_RS0=01KTYXNSJX90Y5E2CQ6SV8K37D
CM4_RS1=01KTWGH2VEZ4DNXXF1H5FTJK1S

submit() {
    local config="$1" ckpt="$2" job="$3"
    if [ -n "$SELECT" ] && [[ "$config" != *"$SELECT"* ]] && [[ "$job" != *"$SELECT"* ]]; then
        return 0
    fi
    local config_path="${CONFIG_DIR}/${config}"
    python -m fme.ace.validate_config --config_type inference "$config_path"
    gantry run \
        --name "$job" \
        --task-name "$job" \
        --description "ACE2S land-feedback inference: ${config%.yaml}" \
        --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
        --workspace ai2/ace \
        --priority normal \
        --not-preemptible \
        --cluster ai2/saturn \
        --cluster ai2/ceres \
        --cluster ai2/jupiter \
        --cluster ai2/prometheus \
        --cluster ai2/titan \
        --env WANDB_USERNAME="$WANDB_USERNAME" \
        --env WANDB_NAME="$job" \
        --env WANDB_JOB_TYPE=inference \
        --env WANDB_RUN_GROUP="$JOB_GROUP" \
        --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
        --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa \
        --dataset-secret google-credentials:/tmp/google_application_credentials.json \
        --dataset "$ckpt:training_checkpoints/best_inference_ckpt.tar:/ckpt.tar" \
        --gpus 1 \
        --shared-memory 50GiB \
        --weka climate-default:/climate-default \
        --budget ai2/atec-climate \
        --system-python \
        --install "pip install --no-deps ." \
        -- python -I -m fme.ace.inference "$config_path"
}

cd "$REPO_ROOT"  # so the config paths resolve regardless of where this is run from

submit pilot-era5.yaml         "$ERA5"    ace2s-lf-pilot-era5
submit frameworkA-era5.yaml    "$ERA5"    ace2s-lf-fwA-era5
submit frameworkA-cm4-rs0.yaml "$CM4_RS0" ace2s-lf-fwA-cm4-rs0
submit frameworkA-cm4-rs1.yaml "$CM4_RS1" ace2s-lf-fwA-cm4-rs1
submit frameworkB-era5.yaml    "$ERA5"    ace2s-lf-fwB-era5
submit frameworkB-cm4-rs0.yaml "$CM4_RS0" ace2s-lf-fwB-cm4-rs0
submit frameworkB-cm4-rs1.yaml "$CM4_RS1" ace2s-lf-fwB-cm4-rs1
