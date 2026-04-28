#!/bin/bash
#
# Submit the CMIP6 daily pilot Argo workflow.
#
# The config YAML should use GCS paths for inventory_path and
# output_directory so that Argo pods can read/write them.  The
# submitting machine also needs GCS access (via gcloud auth) to
# run --dry-run enumeration locally.
#
# Usage:
#   ./submit.sh --config configs/pilot-cloud.yaml --process --stats --normalization
#
# Flags:
#   --config <path>            Path to the process YAML config (required).
#   --inventory-config <path>  Path to the inventory YAML config.  When
#                              provided, inventory.py runs first to
#                              produce/refresh the inventory on GCS.
#   --process                  Run the dataset processing step.
#   --stats                    Run per-dataset stats + presence table.
#   --normalization            Run pooled normalization.
#
# Prerequisites:
#   - argo CLI
#   - yq (https://github.com/mikefarah/yq)
#   - python with the cmip6_data dependencies (for --dry-run enumeration)
#   - gcloud auth (to read GCS inventory during enumeration)
#   - The container image must be built and pushed first:
#       cd scripts/cmip6_data/argo && make build_and_push

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CMIP6_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_PROCESS=false
RUN_STATS=false
RUN_NORMALIZATION=false
INVENTORY_CONFIG=""

while [[ "$#" -gt 0 ]]
do case $1 in
    --config) CONFIG="$2"
    shift;;
    --inventory-config) INVENTORY_CONFIG="$2"
    shift;;
    --process) RUN_PROCESS=true;;
    --stats) RUN_STATS=true;;
    --normalization) RUN_NORMALIZATION=true;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${CONFIG}" ]]; then
    echo "Option --config missing"
    exit 1
fi

if [[ "${RUN_PROCESS}" = false && "${RUN_STATS}" = false && "${RUN_NORMALIZATION}" = false ]]; then
    echo "At least one of --process, --stats, or --normalization must be specified"
    exit 1
fi

ABS_CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"

# Build the inventory on GCS if --inventory-config was given and it
# doesn't already exist.
if [[ -n "${INVENTORY_CONFIG}" ]]; then
    ABS_INV_CONFIG="$(cd "$(dirname "${INVENTORY_CONFIG}")" && pwd)/$(basename "${INVENTORY_CONFIG}")"
    INV_PATH=$(yq -r '.output_path' "${ABS_INV_CONFIG}")
    if python -c "import fsspec; fs, p = fsspec.core.url_to_fs('${INV_PATH}'); exit(0 if fs.exists(p) else 1)" 2>/dev/null; then
        echo "Inventory already exists at ${INV_PATH}, skipping generation."
    else
        echo "Building inventory at ${INV_PATH}..."
        (cd "${CMIP6_DIR}" && python inventory.py --config "${ABS_INV_CONFIG}")
    fi
fi

# Enumerate datasets via process.py --dry-run.  This reads the
# inventory (which may be on GCS) and applies the config's selection
# rules to produce the full task list.
echo "Enumerating datasets from config..."
dataset_lines=$(cd "${CMIP6_DIR}" && python process.py --config "${ABS_CONFIG}" --dry-run)
dataset_keys=()
while IFS=$'\t' read -r source_id experiment variant_label; do
    dataset_keys+=("${source_id}/${experiment}/${variant_label}")
done <<< "${dataset_lines}"

datasets_count=${#dataset_keys[@]}
datasets_count_minus_one=$((datasets_count - 1))

echo "Found ${datasets_count} datasets."

output=$(argo submit "${SCRIPT_DIR}/workflow.yaml" \
    -p run_process="${RUN_PROCESS}" \
    -p run_stats="${RUN_STATS}" \
    -p run_normalization="${RUN_NORMALIZATION}" \
    -p config="$(< "${ABS_CONFIG}")" \
    -p dataset_keys="${dataset_keys[*]}" \
    -p datasets_count_minus_one="${datasets_count_minus_one}")

job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')
echo "Argo job submitted: $job_name"
