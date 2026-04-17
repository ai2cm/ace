#!/bin/bash
set -e

usage() {
    cat <<EOF
Usage: $(basename "$0") GS_PATH [WEKA_PATH]

Submits a Beaker/Gantry job that copies data from a Google Cloud Storage path
to a local Weka directory.

Arguments:
  GS_PATH     The source gs:// path to copy (e.g. gs://vcm-ml-intermediate/data/foo).
  WEKA_PATH   The destination path on Weka (default: /climate-default).
              The GCS directory name will be appended to this path.

Options:
  -h, --help  Show this help message and exit.

Example:
  $(basename "$0") gs://vcm-ml-intermediate/2024-03-01-era5-1deg/train.zarr /climate-default/my-data
EOF
}

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

GS_PATH="$1"
WEKA_PATH="$2"

# Validate gs:// prefix
if [[ "$GS_PATH" != gs://* ]]; then
    echo "Error: GS_PATH must start with gs://"
    exit 1
fi

# Strip trailing slash from paths
GS_PATH="${GS_PATH%/}"
WEKA_PATH="${WEKA_PATH%/}"

REPO_ROOT=$(git rev-parse --show-toplevel)

# Create a job name from the GCS path basename
GS_BASENAME=$(basename "$GS_PATH")
JOB_NAME="gcs-to-weka-${GS_BASENAME}"

cd "$REPO_ROOT" && gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "Copy $GS_PATH to weka at $WEKA_PATH" \
    --docker-image 'google/cloud-sdk:slim' \
    --workspace ai2/climate-titan \
    --priority urgent \
    --not-preemptible \
    --cluster ai2/phobos \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 0 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --no-python \
    --install "echo 'skipping installation step'" \
    -- bash -c "mkdir -p $WEKA_PATH && gsutil -m -o Credentials:gs_service_key_file=/tmp/google_application_credentials.json rsync -r $GS_PATH $WEKA_PATH"
