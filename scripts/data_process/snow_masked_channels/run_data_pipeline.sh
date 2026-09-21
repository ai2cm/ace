#!/bin/bash
# Masked per-land-area snow channels for both daily parents, end to end:
#   stores (era5, cm4) -> stats -> GCS store uploads -> Beaker stats datasets.
# Designed to run detached so it survives the interactive session:
#   nohup caffeinate -i bash run_data_pipeline.sh > <log> 2>&1 &
# Weka copies (scripts/data_process/gcs_to_weka.sh) and training launches are
# done interactively after checking the outputs.

set -e
cd "$(dirname "$0")"
PY=${PY:-python}

ERA5_PARENT=2026-08-07-era5-1deg-8layer-daily-1940-2025
CM4_PARENT=2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily
SUFFIX=land-snow-masked

for ds in era5 cm4; do
  echo "=== $(date) store $ds ==="
  $PY build_masked_snow_channels.py $ds
done
for ds in era5 cm4; do
  echo "=== $(date) stats $ds ==="
  $PY fit_masked_snow_stats.py $ds
done

echo "=== $(date) uploading stores to GCS ==="
gsutil -m rsync -r \
  "store-out/${ERA5_PARENT}-${SUFFIX}.zarr" \
  "gs://vcm-ml-intermediate/${ERA5_PARENT}/${ERA5_PARENT}-${SUFFIX}.zarr"
gsutil -m rsync -r \
  "store-out/${CM4_PARENT}-${SUFFIX}.zarr" \
  "gs://vcm-ml-intermediate/${CM4_PARENT}/${CM4_PARENT}-${SUFFIX}.zarr"

echo "=== $(date) uploading stats datasets to Beaker ==="
beaker dataset create "stats-out/${ERA5_PARENT}-${SUFFIX}-stats" \
  --name "${ERA5_PARENT}-${SUFFIX}-stats-1990-2019" \
  --workspace ai2/ace \
  --desc "ERA5 daily stats plus per-land-area masked snow entries under the _masked names (SWE and cover divided by land_fraction, cover clipped at 1); other variables identical to the 2026-08-07 daily stats"
beaker dataset create "stats-out/${CM4_PARENT}-${SUFFIX}-stats" \
  --name "${CM4_PARENT}-${SUFFIX}-stats" \
  --workspace ai2/ace \
  --desc "CM4 daily stats plus masked snow entries under the _masked names (SWE per land area as stored, cover rescaled from percent to fraction); other variables identical to the 2025-03-21 daily stats"

echo "=== $(date) PIPELINE COMPLETE ==="
