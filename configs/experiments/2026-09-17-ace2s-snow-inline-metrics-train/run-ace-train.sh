#!/bin/bash
# Stage 1: 1-step pre-training of the control and masked-naive arms on both datasets,
# with the anomaly_memory and snow_season aggregators in the inline inference.
#
#   ./run-ace-train.sh            # all four arms
#   ./run-ace-train.sh era5-masked-naive cm4-masked-naive   # a subset
#
# The masked-naive job names carry "-land-snow" because these configs read the
# per-land-area snow channels; the first masked-naive runs, on the per-cell-area
# ERA5 data, kept the plain names (see README).

source "$(dirname "$0")/launch-common.sh"

job_name_for() {
  case "$1" in
    cm4-control)       echo "ace2s-snowmetrics-cm4-daily-control-1-step-pretrain-rs0" ;;
    cm4-masked-naive)  echo "ace2s-snowmetrics-cm4-daily-masked-naive-land-snow-1-step-pretrain-rs0" ;;
    era5-control)      echo "ace2s-snowmetrics-era5-daily-control-1-step-pretrain-rs0" ;;
    era5-masked-naive) echo "ace2s-snowmetrics-era5-daily-masked-naive-land-snow-1-step-pretrain-rs0" ;;
    *) return 1 ;;
  esac
}

ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=(cm4-control cm4-masked-naive era5-control era5-masked-naive)
fi
for arm in "${ARMS[@]}"; do
  job_name_for "$arm" >/dev/null || {
    echo "unknown arm $arm; choose from cm4-control cm4-masked-naive era5-control era5-masked-naive" >&2
    exit 1
  }
done

for arm in "${ARMS[@]}"; do
  run_training "$arm-1-step-pretrain-daily.yaml" "$(job_name_for "$arm")" "seed=0"
done
