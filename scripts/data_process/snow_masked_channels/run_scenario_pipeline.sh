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
#
#   bash run_scenario_pipeline.sh          # stores, then stats
#   bash run_scenario_pipeline.sh stores   # stores and their uploads only
#   bash run_scenario_pipeline.sh stats    # pooled stats and Beaker datasets only
#
# A store that is already complete is skipped by the builder and re-synced as a
# no-op. Designed to run detached so it survives the interactive session:
#   nohup caffeinate -i bash run_scenario_pipeline.sh > <log> 2>&1 &
# Weka copies (scripts/data_process/gcs_to_weka.sh) and training launches are
# done interactively after checking the outputs.

set -eo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
STAGE=${1:-all}
# The training configs mount these stats datasets by name, so the date is fixed.
DATE=${DATE:-2026-10-07}
SUFFIX=land-snow-masked
BUCKET=gs://vcm-ml-intermediate
PIC=2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily
PCT=2026-06-19-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr-daily
RANDCO2_DIR=${BUCKET}/2026-06-19-CM4-like-AM4-random-CO2-1deg-daily
# Per-scenario daily time means over each parent's training window, added to
# every stats dataset as the inline-inference references (the pooled
# time-mean.nc mixes the scenarios' climates).
PIC_TIME_MEAN=${BUCKET}/${PIC}-stats-0156-0236/${PIC}/time-mean.nc
PCT_TIME_MEAN=${BUCKET}/${PCT}-stats-0046-0126/${PCT}/time-mean.nc

case "$STAGE" in
  all|stores|stats) ;;
  *) echo "usage: $0 [all|stores|stats]" >&2; exit 1 ;;
esac

if [[ "$STAGE" != stats ]]; then
  echo "=== $(date) store cm4-picontrol-2026 ==="
  $PY build_masked_snow_channels.py cm4-picontrol-2026
  for level in 1xCO2 2xCO2 4xCO2; do
    for ic in 1 2 3; do
      echo "=== $(date) store cm4-randco2-${level}-ic${ic} ==="
      $PY build_masked_snow_channels.py "cm4-randco2-${level}-ic${ic}"
    done
  done

  echo "=== $(date) uploading stores to GCS ==="
  gsutil -m rsync -r "store-out/${PIC}-${SUFFIX}.zarr" \
    "${BUCKET}/${PIC}/${PIC}-${SUFFIX}.zarr"
  for level in 1xCO2 2xCO2 4xCO2; do
    for ic in 0001 0002 0003; do
      member="random-CO2-${level}-ic_${ic}"
      gsutil -m rsync -r "store-out/${member}-${SUFFIX}.zarr" \
        "${RANDCO2_DIR}/${member}-${SUFFIX}.zarr"
    done
  done
fi

dataset_exists() {
  beaker dataset get "${BEAKER_USER}/$1" > /dev/null 2>&1
}

if [[ "$STAGE" != stores ]]; then
  BEAKER_USER=$(beaker account whoami --format json | python3 -c \
    "import json, sys; d = json.load(sys.stdin); print((d[0] if isinstance(d, list) else d)['name'])")
  for set in pic-1pct pic-1pct-randco2; do
    if dataset_exists "${DATE}-cm4-${set}-daily-stats" && \
       dataset_exists "${DATE}-cm4-${set}-daily-${SUFFIX}-stats"; then
      echo "=== $(date) stats datasets for $set already on Beaker; skipping ==="
      continue
    fi
    echo "=== $(date) pooled stats $set ==="
    pooled=$($PY pool_daily_stats.py "$set" --date "$DATE" | tail -1)
    echo "=== $(date) masked-snow stats $set from $pooled ==="
    $PY fit_masked_snow_stats.py --pool "$set" --parent-stats "$pooled"
    control="stats-out/cm4-${set}-daily-stats"
    treatment="stats-out/cm4-${set}-daily-${SUFFIX}-stats"
    mkdir -p "$control"
    gsutil -m -q cp "${pooled}/*.nc" "$control/"
    for dir in "$control" "$treatment"; do
      gsutil -q cp "$PIC_TIME_MEAN" "$dir/time-mean-piControl.nc"
      gsutil -q cp "$PCT_TIME_MEAN" "$dir/time-mean-1pctCO2.nc"
    done
    echo "=== $(date) uploading stats datasets to Beaker ($set) ==="
    beaker dataset create "$control" \
      --name "${DATE}-cm4-${set}-daily-stats" \
      --workspace ai2/ace \
      --desc "CM4 daily stats pooled over the ${set} scenario-training windows (control arms), plus per-scenario time means time-mean-piControl.nc and time-mean-1pctCO2.nc"
    beaker dataset create "$treatment" \
      --name "${DATE}-cm4-${set}-daily-${SUFFIX}-stats" \
      --workspace ai2/ace \
      --desc "CM4 daily stats pooled over the ${set} scenario-training windows plus masked snow entries under the _masked names (treatment arms), plus per-scenario time means"
  done
fi

echo "=== $(date) PIPELINE STAGE ${STAGE} COMPLETE ==="
