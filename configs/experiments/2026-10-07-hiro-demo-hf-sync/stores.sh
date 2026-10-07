#!/bin/bash
# Sync map for the HiRO demo zarrs -> hf://buckets/allenai/ai2cm-downscaling.
# Sourced by sync.sh and verify.sh; not meant to be run directly.
#
# Destination folders are required by the ace-viz paper-deploy catalog
# (ai2cm/ace-viz#71): hirov2/, hirov1/ and xshield/ (shared by v1 and v2).
#
# Do NOT add /usrhome/hiro-demo-zarrs/2026-09-29-hirov1-global/xshield_100km_full.zarr;
# it is not in the catalog.

# Weka location of what the sync instructions call /usrhome.
USRHOME=${USRHOME:-/climate-default/home/andrep}

HF_BUCKET=allenai/ai2cm-downscaling

HIROV2_DIR=${USRHOME}/hiro-demo-zarrs/2026-08-04-hirov2-global
HIROV1_DIR=${USRHOME}/hiro-demo-zarrs/2026-09-29-hirov1-global

# Row number | weka source directory | HF folder | store name.
# The source path always ends in the store name, so each job copies exactly one store.
STORES=(
    "1|${HIROV2_DIR}|hirov2|2026-08-05-global-hirov2-ace2s-2023.zarr"
    "2|${HIROV2_DIR}|hirov2|2026-08-03-global-hirov2-pp-2023.zarr"
    "3|${HIROV1_DIR}|hirov1|output_6hourly_predictions_ic0000.zarr"
    "4|${HIROV1_DIR}|hirov1|global_hiro_ace_2023.zarr"
    "5|${HIROV1_DIR}|hirov1|global_hiro_perfect_2023.zarr"
    "6|${HIROV1_DIR}|xshield|xshield_100km_2023.zarr"
    "7|${HIROV2_DIR}|hirov2|output_6hourly_predictions_ic0000_2023-onward-rechunked.zarr"
    "8|${USRHOME}|xshield|2026-09-29-x-shield-3km-2023-only.zarr"
)

# Print the STORES rows whose row number or store name matches one of the
# arguments, or every row when no arguments are given. Fails on unknown selectors.
select_stores() {
    if [ "$#" -eq 0 ]; then
        printf '%s\n' "${STORES[@]}"
        return
    fi
    local selector entry row store found
    for selector in "$@"; do
        found=0
        for entry in "${STORES[@]}"; do
            row=${entry%%|*}
            store=${entry##*|}
            if [ "${selector}" = "${row}" ] || [ "${selector}" = "${store}" ]; then
                echo "${entry}"
                found=1
            fi
        done
        if [ "${found}" = "0" ]; then
            echo "Unknown store selector: ${selector}" >&2
            return 1
        fi
    done
}

# Keep Beaker job names to alphanumerics and dashes.
job_name_for() {
    local prefix="$1" store="$2"
    echo "${prefix}-${store%.zarr}" | tr -c 'A-Za-z0-9\n-' '-'
}
