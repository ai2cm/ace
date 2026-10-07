#!/bin/bash
# Launch a gantry job that checks each synced store against its weka source:
#   - the source root zarr.json contains consolidated_metadata
#   - the HF file count and total bytes match the source
#   - with PUBLIC=1, the public zarr.json URL returns 200 with consolidated_metadata
#
# Usage: [PUBLIC=1] ./verify.sh [row number or store name ...]   (default: all rows)

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=stores.sh
source "${SCRIPT_DIR}/stores.sh"

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "${REPO_ROOT}"

PUBLIC=${PUBLIC:-0}

# Arguments to the job are: bucket, PUBLIC flag, then triples of
# (source, folder, store). Prints one PASS/FAIL line per check and exits
# nonzero if any check fails.
VERIFY_COMMAND='
bucket="$1"; public="$2"; shift 2
failed=0
check() {
    if [ "$1" = "0" ]; then echo "  PASS: $2"; else echo "  FAIL: $2"; failed=1; fi
}
while [ "$#" -gt 0 ]; do
    source="$1"; folder="$2"; store="$3"; shift 3
    echo "== ${folder}/${store}"
    grep -q consolidated_metadata "${source}/zarr.json"
    check $? "source zarr.json has consolidated_metadata"
    read -r source_files source_bytes < <(
        python3 -c "
import os, sys
paths = [os.path.join(d, f) for d, _, fs in os.walk(sys.argv[1]) for f in fs]
print(len(paths), sum(os.path.getsize(p) for p in paths))
" "${source}"
    )
    read -r hf_files hf_bytes < <(
        hf buckets list "${bucket}/${folder}/${store}" -R --json | python3 -c "
import json, sys
files = [e for e in json.load(sys.stdin) if e[\"type\"] == \"file\"]
print(len(files), sum(e[\"size\"] for e in files))
"
    )
    [ "${source_files}" = "${hf_files}" ]
    check $? "file count source=${source_files} hf=${hf_files}"
    [ "${source_bytes}" = "${hf_bytes}" ]
    check $? "total bytes source=${source_bytes} hf=${hf_bytes}"
    if [ "${public}" = "1" ]; then
        url="https://huggingface.co/buckets/${bucket}/resolve/${folder}/${store}/zarr.json"
        status=$(curl -sL -o /tmp/zarr.json -w "%{http_code}" "${url}")
        [ "${status}" = "200" ] && grep -q consolidated_metadata /tmp/zarr.json
        check $? "public ${url} status=${status} with consolidated_metadata"
    fi
done
exit "${failed}"
'

job_args=()
selected=$(select_stores "$@")
while IFS='|' read -r row source_dir folder store; do
    job_args+=("${source_dir}/${store}" "${folder}" "${store}")
done <<< "${selected}"

gantry run \
    --name "hf-verify-hiro-demo-$(date +%Y%m%d-%H%M%S)" \
    --description "Verify HiRO demo stores synced to ${HF_BUCKET}" \
    --beaker-image spencerc/hf-cli-gantry \
    --workspace ai2/ace \
    --priority high \
    --cluster ai2/phobos \
    --env-secret HF_TOKEN=hugging-face-token \
    --gpus 0 \
    --no-python \
    --allow-dirty \
    --weka climate-default:/climate-default \
    -- bash -c "${VERIFY_COMMAND}" verify "${HF_BUCKET}" "${PUBLIC}" "${job_args[@]}"
