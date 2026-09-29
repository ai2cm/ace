#!/bin/bash
# Build the ACE-flux forcing zarr from a coupled rollout that saved its atmosphere forcings (CPU job).
#   COUPLED_DS=<result dataset of the saveflux rollout> NAME=samudra-anchor4deg-ace-forcing-<tag> ./launch-4deg-build-ace-forcing.sh
set -euo pipefail
COUPLED_DS="${COUPLED_DS:?}"; NAME="${NAME:?}"
STORE="${STORE:-/climate-default/2026-07-22-cm4-picontrol-4deg-coupled-ocean.zarr}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
cd "$REPO_ROOT"
gantry run --name "$NAME" --task-name "$NAME" --description "ACE-flux ocean forcing zarr from $COUPLED_DS" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority high --preemptible --cluster ai2/ceres --cluster ai2/jupiter --cluster ai2/titan \
  --weka climate-default:/climate-default --dataset "${COUPLED_DS}:/coupled" \
  --cpus 8 --memory 64GiB --budget ai2/atec-climate --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python -I "${SCRIPT_PATH}/tools/build_ace_forcing.py" --coupled /coupled/atmosphere/autoregressive_predictions.nc --store "$STORE" --out /results/ace_forcing.zarr 2>&1 | tail -2
