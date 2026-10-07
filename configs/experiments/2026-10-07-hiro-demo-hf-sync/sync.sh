#!/bin/bash
# Launch one gantry job per store that syncs it from weka to the HF bucket.
#
# Usage: ./sync.sh [row number or store name ...]   (default: all rows in stores.sh)
#
# Examples:
#   ./sync.sh 6                                   # just xshield_100km_2023.zarr
#   IGNORE_EXISTING=1 ./sync.sh 1 2               # resume rows 1 and 2
#   HF_XET_HIGH_PERFORMANCE=0 ./sync.sh 8         # if large shards time out

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=stores.sh
source "${SCRIPT_DIR}/stores.sh"

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"

BEAKER_IMAGE=spencerc/hf-cli-gantry

# HF_XET_HIGH_PERFORMANCE maximizes upload parallelism (see
# scripts/upload_to_hugging_face/Dockerfile), but for stores containing multiple
# large (>4GB) files this can cause upload timeouts. Set to 0 to disable it in
# that case. Overrides the default baked into the image.
HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}

# Set to 1 to pass --ignore-existing, which skips files already present at the
# destination rather than re-uploading them. Useful for resuming a failed sync.
IGNORE_EXISTING=${IGNORE_EXISTING:-0}
SYNC_ARGS=()
if [ "${IGNORE_EXISTING}" = "1" ]; then
    SYNC_ARGS+=(--ignore-existing)
fi

# Never pass --delete: other stores share these destination folders.
# The job refuses to run if the source isn't a zarr store, so a wrong path
# can't upload an empty or wrong directory.
SYNC_COMMAND='
set -e
source="$1"; destination="$2"; shift 2
if [ ! -f "${source}/zarr.json" ]; then
    echo "Missing ${source}/zarr.json; refusing to sync." >&2
    exit 1
fi
echo "Syncing ${source} -> ${destination}"
hf buckets sync "$@" "${source}" "${destination}"
'

selected=$(select_stores "$@")
while IFS='|' read -r row source_dir folder store; do
    source="${source_dir}/${store}"
    destination="hf://buckets/${HF_BUCKET}/${folder}/${store}"
    job_name="$(job_name_for hf-sync "${store}")-$(date +%Y%m%d-%H%M%S)"
    echo "Row ${row}: ${source} -> ${destination} (job ${job_name})"
    gantry run \
        --name "${job_name}" \
        --description "Sync HiRO demo store ${store} from weka to ${HF_BUCKET}/${folder}" \
        --beaker-image "${BEAKER_IMAGE}" \
        --workspace ai2/ace \
        --priority high \
        --cluster ai2/phobos \
        --env-secret HF_TOKEN=hugging-face-token \
        --env HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE}" \
        --gpus 0 \
        --shared-memory 64GiB \
        --min-runtime 8h \
        --no-python \
        --allow-dirty \
        --weka climate-default:/climate-default \
        -- bash -c "${SYNC_COMMAND}" sync "${source}" "${destination}" "${SYNC_ARGS[@]}"
done <<< "${selected}"
