#!/bin/bash
# Masked per-land-area snow channels and pooled stats for the CM4 scenario
# training (piControl + 1pctCO2, and + random-CO2), end to end:
#   stores for the 2026-06-19 piControl daily parent and the nine random-CO2
#   members -> GCS store uploads -> pooled per-store stats per source set ->
#   masked-snow stats per source set -> Beaker stats datasets.
# The 1pctCO2 store already exists (built 2026-09-30). Prerequisites: the daily
# parents and their windowed daily stats from the data pipeline (Makefile targets
# cm4_picontrol_atmosphere_1deg_8layer_200yr_daily,
# cm4_like_am4_random_co2_ensemble_atmosphere_1deg_daily and
# cm4_1pctCO2_atmosphere_1deg_8layer_140yr_daily_stats).
# Designed to run detached so it survives the interactive session:
#   nohup caffeinate -i bash run_scenario_pipeline.sh > <log> 2>&1 &
# Weka copies (scripts/data_process/gcs_to_weka.sh) and training launches are
# done interactively after checking the outputs.

set -e
cd "$(dirname "$0")"
PY=${PY:-python}
DATE=${DATE:-$(date +%F)}
SUFFIX=land-snow-masked
PIC=2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily
RANDCO2_DIR=gs://vcm-ml-intermediate/2026-06-19-CM4-like-AM4-random-CO2-1deg-daily

members=()
for level in 1xCO2 2xCO2 4xCO2; do
  for ic in 1 2 3; do
    members+=("cm4-randco2-${level}-ic${ic}")
  done
done

echo "=== $(date) store cm4-picontrol-2026 ==="
$PY build_masked_snow_channels.py cm4-picontrol-2026
for key in "${members[@]}"; do
  echo "=== $(date) store $key ==="
  $PY build_masked_snow_channels.py "$key"
done

echo "=== $(date) uploading stores to GCS ==="
gsutil -m rsync -r "store-out/${PIC}-${SUFFIX}.zarr" \
  "gs://vcm-ml-intermediate/${PIC}/${PIC}-${SUFFIX}.zarr"
for level in 1xCO2 2xCO2 4xCO2; do
  for ic in 0001 0002 0003; do
    member="random-CO2-${level}-ic_${ic}"
    gsutil -m rsync -r "store-out/${member}-${SUFFIX}.zarr" \
      "${RANDCO2_DIR}/${member}-${SUFFIX}.zarr"
  done
done

for set in pic-1pct pic-1pct-randco2; do
  echo "=== $(date) pooled stats $set ==="
  pooled=$($PY pool_daily_stats.py "$set" --date "$DATE" | tail -1)
  echo "=== $(date) masked-snow stats $set from $pooled ==="
  $PY fit_masked_snow_stats.py --pool "$set" --parent-stats "$pooled"
  echo "=== $(date) uploading stats datasets to Beaker ($set) ==="
  mkdir -p "stats-out/cm4-${set}-daily-stats"
  gsutil -m -q cp "${pooled}/*.nc" "stats-out/cm4-${set}-daily-stats/"
  beaker dataset create "stats-out/cm4-${set}-daily-stats" \
    --name "${DATE}-cm4-${set}-daily-stats" \
    --workspace ai2/ace \
    --desc "CM4 daily stats pooled over the ${set} scenario-training windows (control arms)"
  beaker dataset create "stats-out/cm4-${set}-daily-${SUFFIX}-stats" \
    --name "${DATE}-cm4-${set}-daily-${SUFFIX}-stats" \
    --workspace ai2/ace \
    --desc "CM4 daily stats pooled over the ${set} scenario-training windows plus masked snow entries under the _masked names (treatment arms)"
done

echo "=== $(date) PIPELINE COMPLETE ==="
