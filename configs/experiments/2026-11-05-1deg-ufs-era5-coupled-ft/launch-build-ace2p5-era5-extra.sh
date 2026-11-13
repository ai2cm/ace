#!/bin/bash
# CPU Beaker job that builds the ACE2.5-on-ERA5 extra store (build_ace2p5_era5_extra.py) into its results dataset.
#   ./launch-build-ace2p5-era5-extra.sh   -> mount the results dataset at /extra for launch-1deg-era5-ace2p5-ft.sh
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
cd "$REPO_ROOT"
JOB="build-ace2p5-era5-extra-store"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "Build carbon_dioxide (ppmv), CM4 lake_fraction and sign-flipped surface wind stresses on the ERA5 1deg 1940-2025 time axis for the ACE2.5 ERA5 fine-tune" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority high --min-runtime 1h \
  $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
  --weka climate-default:/climate-default \
  --gpus 0 --cpus 16 --memory 96GiB --shared-memory 32GiB --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python "${SCRIPT_PATH}/build_ace2p5_era5_extra.py" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
