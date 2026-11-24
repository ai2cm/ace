#!/bin/bash
# CPU Beaker job: ingest one period of raw UFS replay cycles (pipeline/raw_ingest.py) into a NEW
# 6-hourly store on weka, resumable (finished timesteps are skipped on restart).
#   YEAR=1985 ./launch-raw-ingest.sh
#   START=1980-01-01T00 END=1980-01-01T18 LABEL=smoke OUT_DIR=/climate-default/<dir> ./launch-raw-ingest.sh
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=${SCRIPT_DIR#$REPO_ROOT/}
YEAR="${YEAR:-}"
START="${START:-${YEAR:?YEAR or START/END}-01-01T00}"
END="${END:-${YEAR}-12-31T18}"
LABEL="${LABEL:-$YEAR}"
OUT_DIR="${OUT_DIR:-/climate-default/2026-11-24-ufs-replay-ocean-1deg-19level-1980-1993-6h}"
WORKERS="${WORKERS:-14}"
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-200GiB}"
MIN_RUNTIME="${MIN_RUNTIME:-3h}"
CLUSTERS="${CLUSTERS:-ai2/titan ai2/jupiter ai2/ceres}"
cd "$REPO_ROOT"
JOB="ufs-replay-raw-ingest-1deg-${LABEL}"
CMD="set -e; export ESMFMKFILE=\$CONDA_PREFIX/lib/esmf.mk; cd ${SCRIPT_PATH}/pipeline; mkdir -p /tmp/work /tmp/scratch ${OUT_DIR}; python raw_ingest.py setup /tmp/work; python raw_ingest.py run /tmp/work ${OUT_DIR}/${LABEL}.zarr ${START} ${END} --workers ${WORKERS} --scratch /tmp/scratch"
out=$(gantry run --name "$JOB" --task-name "$JOB" \
  --description "UFS replay raw-cycle ingest (S3 per-cycle MOM6 + FV3 files, NOAA-method tripolar regrid, ufs-replay-pipeline processing) ${START}..${END} -> ${OUT_DIR}/${LABEL}.zarr" \
  --workspace ai2/ace --priority high --min-runtime "$MIN_RUNTIME" \
  $(for X in $CLUSTERS; do printf -- "--cluster %s " "$X"; done) \
  --weka climate-default:/climate-default \
  --gpus 0 --cpus "$CPUS" --memory "$MEMORY" --shared-memory 16GiB --budget ai2/atec-climate \
  --allow-dirty --python-manager conda --conda-file "${SCRIPT_PATH}/environment-raw-ingest.yaml" --install "echo no project install" \
  -- bash -c "$CMD" 2>&1) || true
echo "$out" | grep -qm1 "beaker.org/ex/" && echo "launched $JOB: $(echo "$out" | grep -om1 'https://beaker.org/ex/[A-Z0-9]*')" || { echo "FAILED $JOB: $(echo "$out" | tail -5)"; exit 1; }
