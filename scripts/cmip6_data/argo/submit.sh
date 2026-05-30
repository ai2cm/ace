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
#   --migrate                  Scan ``output_directory`` for sidecars
#                              whose stored ``schema_version`` is older
#                              than the image's ``SCHEMA_VERSION`` and
#                              spawn one migrate-dataset pod per match.
#                              Each pod chains whatever migration steps
#                              are needed (e.g. 0.1.0 → 0.2.0 → 0.3.0).
#   --augment-only             Together with --process-esgf, drops
#                              ESGF tasks that don't already have a
#                              sidecar on GCS. Those would otherwise
#                              go through process_one_esgf (full
#                              fresh process), which is OOM-prone at
#                              prod scale. Use when iterating on
#                              augment logic without paying for
#                              fresh-mode runs.
#   --force-inventory          Rebuild ``inventory.csv`` /
#                              ``inventory_esgf.csv`` even when they
#                              already exist on GCS. Use when the
#                              inventory queries in config.py have
#                              changed (e.g. new table_ids added).
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
RUN_MIGRATE=false
FORCE_INVENTORY=false
AUGMENT_ONLY=false
STATS_SOURCE_IDS=""
STATS_FORCE=false
STATS_WORKERS="4"
STATS_DATASET_KEYS=""
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
    --migrate) RUN_MIGRATE=true;;
    --augment-only) AUGMENT_ONLY=true;;
    --force-inventory) FORCE_INVENTORY=true;;
    --stats-source-ids) STATS_SOURCE_IDS="$2"; shift;;
    --stats-force) STATS_FORCE=true;;
    --stats-workers) STATS_WORKERS="$2"; shift;;
    --stats-dataset-keys)
        # Space-separated "source_id/experiment/variant_label" triples.
        # Switches the stats stage to per-dataset pods (one
        # compute-stats-dataset pod per triple) instead of the
        # single bucket-wide compute-stats pod. Use for targeted
        # regens where the in-pod worker pool would OOM.
        STATS_DATASET_KEYS="$2"; shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

if [[ "${RUN_PROCESS}" = false && "${RUN_PROCESS_ESGF}" = false \
    && "${RUN_STATS}" = false && "${RUN_NORMALIZATION}" = false \
    && "${RUN_STAGE_EXTERNALS}" = false && "${RUN_MIGRATE}" = false ]]; then
    echo "At least one of --process, --process-esgf, --stats, --normalization, --stage-externals, or --migrate must be specified"
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
# with the given config. Skips the build if the file already exists,
# unless --force-inventory was passed.
build_inventory_if_missing() {
    local inv_config="$1"
    local builder_script="$2"
    local abs_inv_config="$(cd "$(dirname "${inv_config}")" && pwd)/$(basename "${inv_config}")"
    local inv_path
    inv_path=$(yq -r '.output_path' "${abs_inv_config}")
    local exists
    if python -c "import fsspec; fs, p = fsspec.core.url_to_fs('${inv_path}'); exit(0 if fs.exists(p) else 1)" 2>/dev/null; then
        exists=true
    else
        exists=false
    fi
    if [[ "${exists}" = true && "${FORCE_INVENTORY}" != true ]]; then
        echo "Inventory already exists at ${inv_path}, skipping generation."
    else
        if [[ "${exists}" = true ]]; then
            echo "--force-inventory set; rebuilding ${inv_path} via ${builder_script}..."
        else
            echo "Building inventory at ${inv_path} via ${builder_script}..."
        fi
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

    # --augment-only: drop ESGF datasets that don't already have a sidecar
    # on GCS. Those would go through process_one_esgf (fresh full process)
    # which is heavy and OOM-prone at prod scale; this flag is the
    # escape hatch when we just want to augment what we already have.
    if [[ "${AUGMENT_ONLY}" = true ]]; then
        echo "Augment-only: filtering ESGF tasks to those with existing sidecars..."
        filtered_keys=$(cd "${CMIP6_DIR}" && python -c "
import sys
import fsspec
from config import ESGFProcessConfig

cfg = ESGFProcessConfig.from_file('${ABS_ESGF_CONFIG}')
out_dir = cfg.output_directory.rstrip('/')
keys = '''${esgf_dataset_keys[*]}'''.split()
fs, _ = fsspec.core.url_to_fs(out_dir)
for k in keys:
    sidecar = f'{out_dir}/{k}/data.zarr/metadata.json'
    rel = fsspec.core.url_to_fs(sidecar)[1]
    if fs.exists(rel):
        print(k)
")
        esgf_dataset_keys=()
        while IFS= read -r k; do
            [[ -n "${k}" ]] && esgf_dataset_keys+=("${k}")
        done <<< "${filtered_keys}"
        echo "After augment-only filter: ${#esgf_dataset_keys[@]} ESGF datasets."
    fi
fi
esgf_datasets_count=${#esgf_dataset_keys[@]}
esgf_datasets_count_minus_one=$((esgf_datasets_count - 1))

# Migrate: scan the bucket's sidecars and build the list of datasets
# whose stored schema_version is older than the image's SCHEMA_VERSION.
# One pod will migrate each.
migrate_dataset_keys=()
if [[ "${RUN_MIGRATE}" = true ]]; then
    echo "Scanning sidecars for migration-needed datasets..."
    migrate_lines=$(cd "${CMIP6_DIR}" && python -c "
import sys
import fsspec, json
from config import ProcessConfig
from schema_version import SCHEMA_VERSION, version_lt

cfg = ProcessConfig.from_file('${ABS_CONFIG}')
out_dir = cfg.output_directory.rstrip('/')
fs, rel = fsspec.core.url_to_fs(out_dir)
for path in fs.glob(f'{rel}/**/data.zarr/metadata.json'):
    with fs.open(path, 'r') as f:
        try:
            sc = json.load(f)
        except Exception:
            continue
    if not all(k in sc for k in ('source_id', 'experiment', 'variant_label')):
        continue
    v = sc.get('schema_version', '0.0.0')
    if version_lt(v, SCHEMA_VERSION):
        print(f\"{sc['source_id']}/{sc['experiment']}/{sc['variant_label']}\")
")
    while IFS= read -r key; do
        [[ -n "${key}" ]] && migrate_dataset_keys+=("${key}")
    done <<< "${migrate_lines}"
    echo "Found ${#migrate_dataset_keys[@]} datasets needing migration."
fi
migrate_datasets_count=${#migrate_dataset_keys[@]}
migrate_datasets_count_minus_one=$((migrate_datasets_count - 1))

# Stats fan-out count. Default ``-1`` selects the legacy single-pod
# ``compute-stats`` template; non-empty ``STATS_DATASET_KEYS``
# switches to per-dataset pods via ``compute-stats-dataset``.
if [[ -n "${STATS_DATASET_KEYS}" ]]; then
    # shellcheck disable=SC2206
    stats_dataset_keys_arr=(${STATS_DATASET_KEYS})
    STATS_DATASETS_COUNT_MINUS_ONE=$((${#stats_dataset_keys_arr[@]} - 1))
    echo "Stats fan-out: ${#stats_dataset_keys_arr[@]} dataset(s) via compute-stats-dataset"
else
    STATS_DATASETS_COUNT_MINUS_ONE=-1
fi

# Submit. ESGF / migrate params are passed even when their gates are
# false; the templates' ``when`` conditions skip unused steps.
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
    -p run_migrate="${RUN_MIGRATE}" \
    -p config="$(< "${ABS_CONFIG}")" \
    -p dataset_keys="${dataset_keys[*]}" \
    -p datasets_count_minus_one="${datasets_count_minus_one}" \
    -p esgf_config="${ESGF_CONFIG_CONTENT}" \
    -p esgf_inventory_path="${ESGF_INVENTORY_PATH}" \
    -p esgf_dataset_keys="${esgf_dataset_keys[*]}" \
    -p esgf_datasets_count_minus_one="${esgf_datasets_count_minus_one}" \
    -p migrate_dataset_keys="${migrate_dataset_keys[*]}" \
    -p stats_source_ids="${STATS_SOURCE_IDS}" \
    -p stats_force="${STATS_FORCE}" \
    -p stats_workers="${STATS_WORKERS}" \
    -p stats_dataset_keys="${STATS_DATASET_KEYS}" \
    -p stats_datasets_count_minus_one="${STATS_DATASETS_COUNT_MINUS_ONE}" \
    -p migrate_datasets_count_minus_one="${migrate_datasets_count_minus_one}")

job_name=$(echo "$output" | grep 'Name:' | awk '{print $2}')
echo "Argo job submitted: $job_name"
