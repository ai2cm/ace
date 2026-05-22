#!/bin/bash
#
# Submit a Beaker/Gantry job that mirrors a CMIP6 zarr dataset from GCS
# to Weka, converting each per-dataset zarr to yearly netCDF files in
# the process.
#
# Modelled after ``scripts/data_process/copy_zarr_to_weka.sh``: uses
# the repo's deps-bearing beaker image so ``zarr_to_netcdf.py`` has
# xarray / zarr / gcsfs / netcdf4 available, and mounts the Weka
# climate-default share at the same path.
#
# Usage:
#   ./zarr_to_netcdf_beaker.sh --config configs/zarr-to-netcdf-pilot.yaml
#
# Config (YAML):
#   gcs_source:        gs://bucket/<project>/<version>     # required
#   weka_destination:  /climate-default/<project>/<version>  # required
#   workers:           8                                     # optional, default 4
#
# Prerequisites:
#   - gantry on PATH
#   - yq on PATH
#   - ${REPO_ROOT}/latest_deps_only_image.txt points at a beaker image
#     with the python deps installed (set up by the ace repo).

set -e

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift;;
        *) echo "Unknown parameter: $1"; exit 1;;
    esac
    shift
done

if [[ -z "${CONFIG}" ]]; then
    echo "Option --config missing"
    exit 1
fi

GCS_SOURCE=$(yq -r '.gcs_source' "${CONFIG}")
WEKA_DEST=$(yq -r '.weka_destination' "${CONFIG}")
WORKERS=$(yq -r '.workers // 4' "${CONFIG}")

if [[ -z "${GCS_SOURCE}" || "${GCS_SOURCE}" == "null" ]]; then
    echo "Config ${CONFIG} missing required key 'gcs_source'"
    exit 1
fi
if [[ -z "${WEKA_DEST}" || "${WEKA_DEST}" == "null" ]]; then
    echo "Config ${CONFIG} missing required key 'weka_destination'"
    exit 1
fi
if [[ "${GCS_SOURCE}" != gs://* ]]; then
    echo "'gcs_source' must start with gs://; got ${GCS_SOURCE}"
    exit 1
fi
if [[ "${WEKA_DEST}" != /climate-default/* ]]; then
    echo "'weka_destination' must start with /climate-default/; got ${WEKA_DEST}"
    exit 1
fi

# Strip trailing slashes.
GCS_SOURCE="${GCS_SOURCE%/}"
WEKA_DEST="${WEKA_DEST%/}"

REPO_ROOT=$(git rev-parse --show-toplevel)

# Job name: append the trailing path segments of the destination so
# multiple versions don't collide.
SUFFIX=$(echo "${WEKA_DEST#/climate-default/}" | tr '/' '-')
JOB_NAME="zarr-to-netcdf-${SUFFIX}"

cd "${REPO_ROOT}" && gantry run \
    --name "${JOB_NAME}" \
    --task-name "${JOB_NAME}" \
    --description "Mirror ${GCS_SOURCE} to ${WEKA_DEST}, converting zarr -> netCDF" \
    --beaker-image "$(cat ${REPO_ROOT}/latest_deps_only_image.txt)" \
    --workspace ai2/ace \
    --priority normal \
    --cluster ai2/phobos \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --gpus 0 \
    --shared-memory 40GiB \
    --weka climate-default:/climate-default \
    --budget ai2/atec-climate \
    --install "pip install --no-deps ." \
    -- bash -c "set -e && mkdir -p ${WEKA_DEST} && python scripts/cmip6_data/zarr_to_netcdf.py ${GCS_SOURCE} ${WEKA_DEST} --workers ${WORKERS}"
