#!/bin/bash
# Upload ERA5 zarr stores to the HuggingFace bucket.
# Run FROM configs/baselines/era5-ace2s-ablations/.
#
# Usage:
#   ./run-upload-era5-hf.sh 4deg-daily          # upload one store
#   ./run-upload-era5-hf.sh 1deg-daily 1deg-6hourly  # upload multiple
#   ./run-upload-era5-hf.sh stats               # upload stats + clim forcing
#
# The 6h store is ~3.2 TB, so launch it separately with a long min-runtime.
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: $0 STORE [STORE ...]"
  echo "Available stores: 4deg-daily, 1deg-daily, 1deg-6hourly,"
  echo "  stats-1deg, stats-4deg,"
  echo "  clim-1deg-daily, clim-1deg-6hourly, clim-4deg-daily"
  echo ""
  echo "Preset groups:"
  echo "  stats    = stats-1deg stats-4deg clim-1deg-daily clim-1deg-6hourly clim-4deg-daily"
  exit 1
fi

# Expand preset groups
STORES=()
for arg in "$@"; do
  case "$arg" in
    stats)
      STORES+=(stats-1deg stats-4deg clim-1deg-daily clim-1deg-6hourly clim-4deg-daily)
      ;;
    *)
      STORES+=("$arg")
      ;;
  esac
done

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_PATH=$(git rev-parse --show-prefix)
cd "$REPO_ROOT"

CONFIG_PATH="${SCRIPT_PATH}upload_era5_to_hf_bucket.py"

# Build the dataset mounts for stats and clim forcing (only if those stores
# are requested). The main zarr stores live on weka.
EXTRA_ARGS=()
for store in "${STORES[@]}"; do
  case "$store" in
    stats-1deg)
      EXTRA_ARGS+=(--dataset "andrep/2026-03-19-era5-1deg-8layer-stats-1990-2019:/statsdata")
      ;;
    stats-4deg)
      EXTRA_ARGS+=(--dataset "andrep/2026-09-08-era5-4deg-8layer-stats-1990-2019:/statsdata-4deg")
      ;;
    clim-1deg-daily)
      EXTRA_ARGS+=(--dataset "01M3778WCW285GX2WQ7WT8PWW6:/clim-forcing-1deg-daily")
      ;;
    clim-1deg-6hourly)
      EXTRA_ARGS+=(--dataset "01M3PX84R361KT2BT30Z1DMA99:/clim-forcing-1deg-6h")
      ;;
    clim-4deg-daily)
      EXTRA_ARGS+=(--dataset "01M3PX831YBH5PNEEKA8R7T70W:/clim-forcing-4deg-daily")
      ;;
  esac
done

JOB_NAME="upload-era5-hf-$(echo "${STORES[*]}" | tr ' ' '-')"

gantry run \
  --name "$JOB_NAME" \
  --description "Upload ERA5 stores to HF bucket: ${STORES[*]}" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority normal \
  --not-preemptible \
  --cluster ai2/jupiter \
  --gpus 0 \
  --cpus 8 \
  --memory 32GiB \
  --env-secret HF_TOKEN=hugging-face-token \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --allow-dirty \
  --system-python \
  --install "pip install --no-deps . && pip install huggingface_hub" \
  "${EXTRA_ARGS[@]}" \
  -- python "$CONFIG_PATH" "${STORES[@]}"
