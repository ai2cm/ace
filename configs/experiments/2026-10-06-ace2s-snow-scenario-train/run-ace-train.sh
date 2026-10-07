#!/bin/bash
# Stage 1: 1-step pre-training of the control and masked-naive arms on the two
# scenario source sets (piControl + 1pctCO2, and the same plus the random-CO2
# ensemble), with carbon_dioxide as a forcing.
#
#   ./run-ace-train.sh                                  # all four arms
#   ./run-ace-train.sh cm4-control-pic-1pct cm4-masked-naive-pic-1pct   # a subset

source "$(dirname "$0")/launch-common.sh"

ALL_ARMS=(cm4-control-pic-1pct cm4-masked-naive-pic-1pct
          cm4-control-pic-1pct-randco2 cm4-masked-naive-pic-1pct-randco2)

ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  ARMS=("${ALL_ARMS[@]}")
fi
for arm in "${ARMS[@]}"; do
  if [[ ! " ${ALL_ARMS[*]} " =~ " $arm " ]]; then
    echo "unknown arm $arm; choose from ${ALL_ARMS[*]}" >&2
    exit 1
  fi
done

for arm in "${ARMS[@]}"; do
  run_training "$arm-1-step-pretrain-daily.yaml" "ace2s-snowscen-${arm}-daily-1-step-pretrain-rs0" "seed=0"
done
