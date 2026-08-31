#!/bin/bash
# CPU job: strided snapshot extraction of atmosphere fields onto the ocean's
# 5-day axis, for the all-in-one arm. Pure indexing, no averaging.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT_PATH="configs/experiments/2026-08-18-samudra-enso-rollout-interventions"
ATMOS=/climate-default/2025-06-18-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr.zarr
OCEAN=/climate-default/2025-10-16-cm4-1pctCO2-140yr-ocean-no-smoothing.zarr
OUT=/climate-default/2026-08-31-cm4-1pctCO2-140yr-atmos-5day-snapshots.zarr

cd "$REPO_ROOT"
gantry run \
  --name samudra-enso-allinone-extract \
  --task-name samudra-enso-allinone-extract \
  --description "All-in-one arm data prep: 5-day instantaneous atmosphere snapshots" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace \
  --priority "${PRIORITY:-normal}" \
  --preemptible \
  --cluster ai2/ceres \
  --cluster ai2/jupiter \
  --cluster ai2/titan \
  --weka climate-default:/climate-default \
  --gpus 0 \
  --budget ai2/atec-climate \
  --allow-dirty \
  --system-python \
  --install "pip install --no-deps ." \
  -- python "${SCRIPT_PATH}/allinone_extract_atmos_snapshots.py" \
       --atmos "$ATMOS" --ocean "$OCEAN" --out "$OUT"
