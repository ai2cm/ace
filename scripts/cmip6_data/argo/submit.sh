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
#   --stage-externals          Run external_forcings.py once to stage
#                              CO2/SO2/BC/forest into
#                              <output>/external_forcings/. Heavy one-time
#                              step (~30 GB source download); re-running
#                              skips already-staged scenarios.
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
RUN_STAGE_EXTERNALS=false
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
    --stage-externals) RUN_STAGE_EXTERNALS=true;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ -z "${CONFIG}" ]]; then
    echo "Option --config missing"
    exit 1
fi

if [[ "${RUN_PROCESS}" = false && "${RUN_STATS}" = false && "${RUN_NORMALIZATION}" = false && "${RUN_STAGE_EXTERNALS}" = false ]]; then
    echo "At least one of --process, --stats, --normalization, or --stage-externals must be specified"
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

# Enumerate datasets via process.py --dry-run when --process is requested.
# (Skip when only running --stage-externals / --stats / --normalization, which
# don't need the per-dataset task list and may not have a GCS inventory yet.)
dataset_keys=()
if [[ "${RUN_PROCESS}" = true ]]; then
    echo "Enumerating datasets from config..."
    dataset_lines=$(cd "${CMIP6_DIR}" && python process.py --config "${ABS_CONFIG}" --dry-run)
    while IFS=$'\t' read -r source_id experiment variant_label; do
        dataset_keys+=("${source_id}/${experiment}/${variant_label}")
    done <<< "${dataset_lines}"
    echo "Found ${#dataset_keys[@]} datasets."
fi

datasets_count=${#dataset_keys[@]}
datasets_count_minus_one=$((datasets_count - 1))

output=$(argo submit "${SCRIPT_DIR}/workflow.yaml" \
    -p run_process="${RUN_PROCESS}" \
    -p run_stats="${RUN_STATS}" \
    -p run_normalization="${RUN_NORMALIZATION}" \
    -p run_stage_externals="${RUN_STAGE_EXTERNALS}" \
    -p config="$(< "${ABS_CONFIG}")" \
    -p dataset_keys="${dataset_keys[*]}" \
    -p datasets_count_minus_one="${datasets_count_minus_one}")

job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')
echo "Argo job submitted: $job_name"
