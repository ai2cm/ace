#!/bin/bash
set -e

usage() {
    cat <<EOF
Usage: $(basename "$0") GS_PATH WEKA_PATH [GS_PATH WEKA_PATH ...]

Submits a Beaker/Gantry job that copies data from one or more Google Cloud
Storage paths to local Weka directories. The pairs are copied in order, in a
single job.

Arguments:
  GS_PATH     The source gs:// path to copy (e.g. gs://vcm-ml-intermediate/data/foo).
  WEKA_PATH   The destination path on Weka (e.g. /climate-default/foo).
              The contents of GS_PATH are synced into this directory.

Options:
  -h, --help  Show this help message and exit.

Examples:
  $(basename "$0") gs://vcm-ml-intermediate/2024-03-01-era5-1deg/train.zarr /climate-default/my-data/train.zarr

  $(basename "$0") \\
    gs://vcm-ml-intermediate/foo.zarr /climate-default/foo.zarr \\
    gs://vcm-ml-intermediate/foo-stats /climate-default/foo-stats
EOF
}

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# % 2 != 0 )); then
    echo "Error: arguments must be GS_PATH WEKA_PATH pairs (got $# arguments)"
    exit 1
fi

TOTAL_PAIRS=$(($# / 2))
COPY_CMD=""
N_PAIRS=0
while [[ $# -gt 0 ]]; do
    GS_PATH="$1"
    WEKA_PATH="$2"
    shift 2

    # Validate gs:// prefix
    if [[ "$GS_PATH" != gs://* ]]; then
        echo "Error: GS_PATH must start with gs:// (got $GS_PATH)"
        exit 1
    fi

    # Strip trailing slash from paths
    GS_PATH="${GS_PATH%/}"
    WEKA_PATH="${WEKA_PATH%/}"

    if (( N_PAIRS == 0 )); then
        FIRST_GS_PATH="$GS_PATH"
        FIRST_WEKA_PATH="$WEKA_PATH"
    fi

    N_PAIRS=$((N_PAIRS + 1))

    # Log a marker before each pair so a failed job shows where it stopped
    COPY_CMD+="${COPY_CMD:+ && }echo '[$N_PAIRS/$TOTAL_PAIRS] $GS_PATH -> $WEKA_PATH'"
    COPY_CMD+=" && mkdir -p $WEKA_PATH && gsutil -m -o Credentials:gs_service_key_file=/tmp/google_application_credentials.json rsync -r $GS_PATH $WEKA_PATH"
done
COPY_CMD+=" && echo 'Done: copied $N_PAIRS/$TOTAL_PAIRS pairs'"

REPO_ROOT=$(git rev-parse --show-toplevel)

# Create a job name from the first GCS path basename
GS_BASENAME=$(basename "$FIRST_GS_PATH")
JOB_NAME="gcs-to-weka-${GS_BASENAME}"
DESCRIPTION="Copy $FIRST_GS_PATH to weka at $FIRST_WEKA_PATH"
if (( N_PAIRS > 1 )); then
    JOB_NAME="${JOB_NAME}-and-$((N_PAIRS - 1))-more"
    DESCRIPTION="Copy $FIRST_GS_PATH and $((N_PAIRS - 1)) more to weka"
fi

cd "$REPO_ROOT" && gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "$DESCRIPTION" \
    --docker-image 'google/cloud-sdk:slim' \
    --workspace ai2/ace \
    --priority urgent \
    --min-runtime 8h \
    --cluster ai2/phobos \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 0 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --no-python \
    --install "echo 'skipping installation step'" \
    -- bash -c "$COPY_CMD"
