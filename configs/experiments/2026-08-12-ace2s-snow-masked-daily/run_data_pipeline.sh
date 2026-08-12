#!/bin/bash
# Full masked-snow data pipeline, designed to run detached (nohup + caffeinate)
# so it survives the interactive session:
#   sidecar zarrs (era5, cm4) -> masked stats fits -> GCS sidecar uploads
#   -> Beaker stats dataset uploads
# Weka copies and training launches are deliberately NOT here (submissions are
# done interactively after verification).
#
# Usage:  nohup caffeinate -i bash run_data_pipeline.sh > <log> 2>&1 &

set -e
cd "$(dirname "$0")"
PY=/opt/homebrew/Caskroom/miniconda/base/envs/fme/bin/python

echo "=== $(date) sidecar era5 ==="
$PY build_masked_snow_channels.py era5
echo "=== $(date) sidecar cm4 ==="
$PY build_masked_snow_channels.py cm4
echo "=== $(date) stats era5 ==="
$PY fit_masked_snow_stats.py era5
echo "=== $(date) stats cm4 ==="
$PY fit_masked_snow_stats.py cm4

echo "=== $(date) uploading sidecars to GCS ==="
gsutil -m rsync -r \
  "sidecar-out/2026-08-07-era5-1deg-8layer-daily-1940-2025-snow-masked.zarr" \
  "gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025/2026-08-07-era5-1deg-8layer-daily-1940-2025-snow-masked.zarr"
gsutil -m rsync -r \
  "sidecar-out/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-masked.zarr" \
  "gs://vcm-ml-intermediate/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-masked.zarr"

echo "=== $(date) uploading stats datasets to Beaker ==="
beaker dataset create stats-out/era5-masked-naive \
  --name 2026-08-07-era5-1deg-8layer-daily-snow-masked-naive-stats-1990-2019 \
  --workspace ai2/ace \
  --desc "ERA5 daily stats with snow entries as masked-domain raw statistics under the _masked names; other variables identical to andrep/2026-08-07-era5-1deg-8layer-daily-stats-1990-2019"
beaker dataset create stats-out/era5-masked-log1p \
  --name 2026-08-07-era5-1deg-8layer-daily-snow-masked-log1p-stats-1990-2019 \
  --workspace ai2/ace \
  --desc "ERA5 daily stats with snow entries as masked-domain log1p/logit statistics under the _masked names; other variables identical to andrep/2026-08-07-era5-1deg-8layer-daily-stats-1990-2019"
beaker dataset create stats-out/cm4-masked-naive \
  --name 2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-masked-naive-stats \
  --workspace ai2/ace \
  --desc "CM4 daily stats with snow entries as masked-domain raw statistics under the _masked names; other variables identical to brianhenn/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-stats"
beaker dataset create stats-out/cm4-masked-log1p \
  --name 2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-masked-log1p-stats \
  --workspace ai2/ace \
  --desc "CM4 daily stats with snow entries as masked-domain log1p/logit statistics under the _masked names; other variables identical to brianhenn/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-stats"

echo "=== $(date) PIPELINE COMPLETE ==="
