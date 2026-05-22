#!/bin/bash
#
# Submit the CMIP6 daily Argo workflow (production or pilot).
#
# The config YAML should use GCS paths for inventory_path and
# output_directory so that Argo pods can read/write them.  The
# submitting machine also needs GCS access (via gcloud auth) to
# run --dry-run enumeration locally.
#
# Usage examples:
#   # Pangeo only, smoke test
#   ./submit.sh --config configs/pilot-cloud.yaml --process --stats
#
#   # Full production run: both pipelines + externals + stats
#   ./submit.sh \
#     --inventory-config configs/cmip6-multimodel-4deg-1940-2100-inventory.yaml \
#     --config configs/cmip6-multimodel-4deg-1940-2100-cloud.yaml \
#     --esgf-inventory-config configs/cmip6-multimodel-4deg-1940-2100-esgf-inventory.yaml \
#     --esgf-config configs/cmip6-multimodel-4deg-1940-2100-esgf-cloud.yaml \
#     --stage-externals --process --process-esgf --stats
#
# Flags:
#   --config <path>            Path to the Pangeo process YAML config.
#                              Required for --process.
#   --inventory-config <path>  Path to the Pangeo inventory YAML config.
#                              When provided, inventory.py runs locally
#                              first to produce/refresh the inventory on
#                              GCS (skipped if already present).
#   --esgf-config <path>       Path to the ESGF process YAML config.
#                              Required for --process-esgf.
#   --esgf-inventory-config <path>
#                              Path to the ESGF inventory YAML config.
#                              When provided, inventory_esgf.py runs
#                              locally first.
#   --process                  Run the Pangeo dataset processing step.
#   --process-esgf             Run the ESGF dataset processing step
#                              (after Pangeo if both are selected).
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
RUN_PROCESS_ESGF=false
RUN_STATS=false
RUN_NORMALIZATION=false
RUN_STAGE_EXTERNALS=false
CONFIG=""
ESGF_CONFIG=""
INVENTORY_CONFIG=""
ESGF_INVENTORY_CONFIG=""

while [[ "$#" -gt 0 ]]
do case $1 in
    --config) CONFIG="$2"
    shift;;
    --inventory-config) INVENTORY_CONFIG="$2"
    shift;;
    --esgf-config) ESGF_CONFIG="$2"
    shift;;
    --esgf-inventory-config) ESGF_INVENTORY_CONFIG="$2"
    shift;;
    --process) RUN_PROCESS=true;;
    --process-esgf) RUN_PROCESS_ESGF=true;;
    --stats) RUN_STATS=true;;
    --normalization) RUN_NORMALIZATION=true;;
    --stage-externals) RUN_STAGE_EXTERNALS=true;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ "${RUN_PROCESS}" = false && "${RUN_PROCESS_ESGF}" = false \
    && "${RUN_STATS}" = false && "${RUN_NORMALIZATION}" = false \
    && "${RUN_STAGE_EXTERNALS}" = false ]]; then
    echo "At least one of --process, --process-esgf, --stats, --normalization, or --stage-externals must be specified"
    exit 1
fi

# --config is required for everything that uses the main config
# (--process, --stage-externals, --stats, --normalization use the
# Pangeo config's output_directory). --process-esgf alone could in
# principle skip it, but in practice we always pair the two so we
# keep the requirement simple.
if [[ -z "${CONFIG}" ]]; then
    echo "Option --config missing"
    exit 1
fi
ABS_CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"

# --esgf-config required if --process-esgf is set.
if [[ "${RUN_PROCESS_ESGF}" = true && -z "${ESGF_CONFIG}" ]]; then
    echo "--process-esgf requires --esgf-config"
    exit 1
fi
ABS_ESGF_CONFIG=""
if [[ -n "${ESGF_CONFIG}" ]]; then
    ABS_ESGF_CONFIG="$(cd "$(dirname "${ESGF_CONFIG}")" && pwd)/$(basename "${ESGF_CONFIG}")"
fi

# Helper: build an inventory CSV on GCS by running the given script
# with the given config if the inventory doesn't already exist there.
build_inventory_if_missing() {
    local inv_config="$1"
    local builder_script="$2"
    local abs_inv_config="$(cd "$(dirname "${inv_config}")" && pwd)/$(basename "${inv_config}")"
    local inv_path
    inv_path=$(yq -r '.output_path' "${abs_inv_config}")
    if python -c "import fsspec; fs, p = fsspec.core.url_to_fs('${inv_path}'); exit(0 if fs.exists(p) else 1)" 2>/dev/null; then
        echo "Inventory already exists at ${inv_path}, skipping generation."
    else
        echo "Building inventory at ${inv_path} via ${builder_script}..."
        (cd "${CMIP6_DIR}" && python "${builder_script}" --config "${abs_inv_config}")
    fi
    # Echo the path so the caller can capture it.
    echo "${inv_path}" > /tmp/last_inventory_path
}

# Pangeo inventory.
if [[ -n "${INVENTORY_CONFIG}" ]]; then
    build_inventory_if_missing "${INVENTORY_CONFIG}" "inventory.py"
fi

# ESGF inventory.
ESGF_INVENTORY_PATH=""
if [[ -n "${ESGF_INVENTORY_CONFIG}" ]]; then
    build_inventory_if_missing "${ESGF_INVENTORY_CONFIG}" "inventory_esgf.py"
    ESGF_INVENTORY_PATH=$(cat /tmp/last_inventory_path)
fi

# Enumerate Pangeo datasets via process.py --dry-run when --process is
# requested. (Skip when only running --stage-externals / --stats /
# --normalization / --process-esgf, none of which need the Pangeo task
# list and may not have a GCS inventory yet.)
dataset_keys=()
if [[ "${RUN_PROCESS}" = true ]]; then
    echo "Enumerating Pangeo datasets from config..."
    dataset_lines=$(cd "${CMIP6_DIR}" && python process.py --config "${ABS_CONFIG}" --dry-run)
    while IFS=$'\t' read -r source_id experiment variant_label; do
        dataset_keys+=("${source_id}/${experiment}/${variant_label}")
    done <<< "${dataset_lines}"
    echo "Found ${#dataset_keys[@]} Pangeo datasets."
fi
datasets_count=${#dataset_keys[@]}
datasets_count_minus_one=$((datasets_count - 1))

# Enumerate ESGF datasets via process_esgf.py --dry-run when
# --process-esgf is requested. process_esgf's dry-run prints
# ``source_id<TAB>experiment<TAB>variant_label<TAB>...extra-cols...``
# — we only need the first three fields.
esgf_dataset_keys=()
if [[ "${RUN_PROCESS_ESGF}" = true ]]; then
    if [[ -z "${ESGF_INVENTORY_PATH}" ]]; then
        echo "--process-esgf needs --esgf-inventory-config to know the inventory location"
        exit 1
    fi
    echo "Enumerating ESGF datasets from config..."
    esgf_lines=$(cd "${CMIP6_DIR}" && python process_esgf.py \
        --config "${ABS_ESGF_CONFIG}" \
        --inventory "${ESGF_INVENTORY_PATH}" \
        --dry-run)
    while IFS=$'\t' read -r source_id experiment variant_label _rest; do
        esgf_dataset_keys+=("${source_id}/${experiment}/${variant_label}")
    done <<< "${esgf_lines}"
    echo "Found ${#esgf_dataset_keys[@]} ESGF datasets."
fi
esgf_datasets_count=${#esgf_dataset_keys[@]}
esgf_datasets_count_minus_one=$((esgf_datasets_count - 1))

# Submit. ESGF params are passed even when --process-esgf is false; the
# template gates on run_process_esgf so unused params are harmless.
ESGF_CONFIG_CONTENT=""
if [[ -n "${ABS_ESGF_CONFIG}" ]]; then
    ESGF_CONFIG_CONTENT="$(< "${ABS_ESGF_CONFIG}")"
fi
output=$(argo submit "${SCRIPT_DIR}/workflow.yaml" \
    -p run_process="${RUN_PROCESS}" \
    -p run_process_esgf="${RUN_PROCESS_ESGF}" \
    -p run_stats="${RUN_STATS}" \
    -p run_normalization="${RUN_NORMALIZATION}" \
    -p run_stage_externals="${RUN_STAGE_EXTERNALS}" \
    -p config="$(< "${ABS_CONFIG}")" \
    -p dataset_keys="${dataset_keys[*]}" \
    -p datasets_count_minus_one="${datasets_count_minus_one}" \
    -p esgf_config="${ESGF_CONFIG_CONTENT}" \
    -p esgf_inventory_path="${ESGF_INVENTORY_PATH}" \
    -p esgf_dataset_keys="${esgf_dataset_keys[*]}" \
    -p esgf_datasets_count_minus_one="${esgf_datasets_count_minus_one}")

job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')
echo "Argo job submitted: $job_name"
