#!/bin/bash
# Rerun the three random-CO2 analyses (per-member metrics + CO2 regressions, pooled-slope figure, Spencer's time-series
# figures + held-out table) into $OUT. The scripts discover every finished before/after evaluation on wandb.
set -euo pipefail
OUT="${OUT:-/home/troya/ace-randco2-figs/analysis}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"; cp "$SCRIPT_DIR/plotting.py" "$OUT/"
export RANDCO2_OUT="$OUT"
cd "$OUT"
python "$SCRIPT_DIR/randco2_compare.py" > compare.log 2>&1
python "$SCRIPT_DIR/randco2_fig.py" > fig.log 2>&1
python "$SCRIPT_DIR/randco2_timeseries.py" > timeseries.log 2>&1
python "$SCRIPT_DIR/randco2_tendency_phase.py" > tendency.log 2>&1
python "$SCRIPT_DIR/randco2_epoch_summary.py" > summary.log 2>&1
echo "refreshed $OUT: $(ls figures | tr '\n' ' ')"
