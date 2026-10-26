#!/bin/bash
# Copy GCS paths to new weka paths in one CPU beaker job, like
# scripts/data_process/gcs_to_weka.sh, but the job fails before copying
# anything if any destination already exists, so no weka data is overwritten.
#
# Usage: bash copy_to_weka_no_clobber.sh GS_PATH WEKA_PATH [GS_PATH WEKA_PATH ...]

set -e

if (( $# < 2 || $# % 2 != 0 )); then
    echo "Usage: $(basename "$0") GS_PATH WEKA_PATH [GS_PATH WEKA_PATH ...]"
    exit 1
fi

GCP_CREDS=/tmp/google_application_credentials.json
CHECK_CMD="true"
COPY_CMD="true"
FIRST_GS_PATH="${1%/}"
while [[ $# -gt 0 ]]; do
    GS_PATH="${1%/}"
    WEKA_PATH="${2%/}"
    shift 2
    if [[ "$GS_PATH" != gs://* || "$WEKA_PATH" != /climate-default/* ]]; then
        echo "Error: expected gs://... -> /climate-default/... (got $GS_PATH -> $WEKA_PATH)"
        exit 1
    fi
    if ! gsutil ls "$GS_PATH/" >/dev/null 2>&1; then
        echo "Error: $GS_PATH does not exist"
        exit 1
    fi
    CHECK_CMD+=" && { [[ ! -e $WEKA_PATH ]] || { echo 'Error: $WEKA_PATH exists'; exit 1; }; }"
    COPY_CMD+=" && echo '$GS_PATH -> $WEKA_PATH' && mkdir -p $WEKA_PATH"
    COPY_CMD+=" && gsutil -m -o Credentials:gs_service_key_file=$GCP_CREDS rsync -r $GS_PATH $WEKA_PATH"
done

JOB_NAME="gcs-to-weka-no-clobber-$(basename "$FIRST_GS_PATH")"
cd "$(git rev-parse --show-toplevel)"
gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "Copy $FIRST_GS_PATH (and any further pairs) to new weka paths" \
    --docker-image 'google/cloud-sdk:slim' \
    --workspace ai2/ace \
    --priority urgent \
    --min-runtime 8h \
    --cluster ai2/phobos \
    --dataset-secret "google-credentials:$GCP_CREDS" \
    --gpus 0 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --no-python \
    --install "echo 'skipping installation step'" \
    -- bash -c "$CHECK_CMD && $COPY_CMD && echo Done"
