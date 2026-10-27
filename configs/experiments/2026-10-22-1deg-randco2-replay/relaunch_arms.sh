#!/bin/bash
# Launch (or relaunch) the two 1deg arms and their per-epoch watchdogs; the before-coupling evals already exist.
# Troy 2026-10-23, after the OOM relaunch (configs now carry activation checkpointing).
set -uo pipefail
ATMOS=01M5JJTWWGYYSACGH3VY6GN6RF
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$(cd "$SCRIPT_DIR/../2026-10-09-4deg-hybrid-ppad/tools" && pwd)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fme
log() { echo "$(date -u +%FT%TZ) $*"; }
# the watchdogs' per-epoch analysis refresh scores the 1deg twin (tools/randco2_compare.py RANDCO2_RES=1deg) into its own dir
export RANDCO2_RES=1deg OUT=/home/troya/ace-randco2-figs/analysis-1deg
cd "$SCRIPT_DIR"
# To continue a previous submission's run, export EXISTING_alt=<results dataset> and/or EXISTING_alt_replay=<...>:
# the launcher then mounts it at /existing-results and uses the -resume config (resume_results + resume_wandb).
for ARM in alt alt-replay; do
  var="EXISTING_${ARM//-/_}"
  EXISTING_RESULTS_DS="${!var:-}" ATMOS_DS="$ATMOS" ARM=$ARM ./launch-1deg-coupled-ft.sh 2>&1 | tail -2
  sleep 20
  beaker experiment get "troya/1deg-randco2rs1-fmean-$ARM" --format json >/dev/null 2>&1 || { log "launch of $ARM failed; no watchdog"; continue; }
  B=1460; [ "$ARM" = alt-replay ] && B=2190
  (BATCHES_PER_EPOCH=$B EXPERIMENT="troya/1deg-randco2rs1-fmean-$ARM" TAG="rco2rs1-1deg-$ARM" WORK="/home/troya/ace-randco2-figs/watchdog-work-1deg-$ARM" EVAL_LAUNCHER="$SCRIPT_DIR/launch-1deg-ace-randco2-evals.sh" EVAL_JOB_PREFIX=ace1deg-randco2 nohup setsid "$TOOLS/watch_coupled_ft_epochs.sh" >> "/home/troya/ace-randco2-figs/watchdog-1deg-$ARM.log" 2>&1 &)
  log "launched $ARM and its watchdog (batches/epoch $B)"
done
log "done"
