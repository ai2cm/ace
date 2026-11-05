#!/bin/bash
# CPU Beaker job that builds the CM4-variable-set twin of the 1deg UFS store on weka (build_cm4vars_store.py); run after the
# weka mirror of the clean store exists (scripts/data_process/gcs_to_weka.sh).
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
cd "$REPO_ROOT"
JOB=build-ufs1deg-cm4vars-store
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "1deg UFS 2026-10-06 store + CM4 placeholder fields (hfgeou, zero sfdsi/UI/VI) on weka" \
  --beaker-image "$(cat "$REPO_ROOT/latest_deps_only_image.txt")" \
  --workspace ai2/ace --priority urgent --min-runtime 4h \
  --cluster ai2/phobos --gpus 0 --shared-memory 40GiB \
  --weka climate-default:/climate-default \
  --budget ai2/atec-climate \
  --allow-dirty --system-python --install "pip install --no-deps ." \
  -- python "${SCRIPT_PATH}/build_cm4vars_store.py" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -3)"; exit 1; }
