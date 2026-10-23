#!/bin/bash
# The rs1 results dataset was still uncommitted when launch_after_rs1.sh fired (every launch failed with "uncommitted
# dataset"). Wait for Beaker to commit it, then do the same launches. Troy 2026-10-22.
set -uo pipefail
ATMOS=01M5JJTWWGYYSACGH3VY6GN6RF
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS="$SCRIPT_DIR/../2026-10-09-4deg-hybrid-ppad/tools"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fme
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for dataset $ATMOS to commit"
until c=$(beaker dataset get $ATMOS --format json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)[0].get('committed',''))"); [ -n "$c" ] && [ "${c:0:4}" != "0001" ]; do sleep 300; done
log "committed $c"
cd "$SCRIPT_DIR"
export CLUSTERS="ai2/rhea ai2/prometheus ai2/ceres ai2/jupiter ai2/titan ai2/saturn ai2/neptune"
CKPT_DS="$ATMOS" TAG=rco2rs1-before ./launch-1deg-ace-randco2-evals.sh 2>&1 | tail -4
unset CLUSTERS
for ARM in alt alt-replay; do
  ATMOS_DS="$ATMOS" ARM=$ARM ./launch-1deg-coupled-ft.sh 2>&1 | tail -2
  sleep 20
  beaker experiment get "troya/1deg-randco2rs1-fmean-$ARM" --format json >/dev/null 2>&1 || { log "launch of $ARM failed; no watchdog"; continue; }
  B=1460; [ "$ARM" = alt-replay ] && B=2190
  (BATCHES_PER_EPOCH=$B EXPERIMENT="troya/1deg-randco2rs1-fmean-$ARM" TAG="rco2rs1-1deg-$ARM" WORK="/home/troya/ace-randco2-figs/watchdog-work-1deg-$ARM" EVAL_LAUNCHER="$SCRIPT_DIR/launch-1deg-ace-randco2-evals.sh" EVAL_JOB_PREFIX=ace1deg-randco2 nohup setsid "$TOOLS/watch_coupled_ft_epochs.sh" >> "/home/troya/ace-randco2-figs/watchdog-1deg-$ARM.log" 2>&1 &)
  log "launched $ARM and its watchdog (batches/epoch $B)"
done
log "done"
