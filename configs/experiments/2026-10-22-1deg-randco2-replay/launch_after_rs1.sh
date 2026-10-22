#!/bin/bash
# Waits for Spencer's 1deg random-CO2 ACE2S rs1 fine-tune to finish, then: scores it on the 1deg random-CO2 ic_0003
# evaluators ("before"), launches the alternating and alternating+replay coupled fine-tunes, and starts a per-epoch
# watchdog for each. Troy 2026-10-22. Run detached: nohup setsid ./launch_after_rs1.sh > /home/troya/ace-randco2-figs/1deg-launcher.log 2>&1 &
set -uo pipefail
RS1=01M55CYWQ60CYEZYRXCQBQE2J2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS="$SCRIPT_DIR/../2026-10-09-4deg-hybrid-ppad/tools"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for the rs1 fine-tune $RS1"
while true; do
  st=$(beaker experiment get $RS1 --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]; j=e['jobs'][-1]; s=j['status']
print(('done '+j['result']['beaker']) if s.get('exited') and s.get('exitCode')==0 else 'running')" 2>/dev/null)
  case "$st" in done*) ATMOS=${st#done }; break;; esac
  sleep 600
done
log "rs1 finished; results $ATMOS"
cd "$SCRIPT_DIR"
export CLUSTERS="ai2/rhea ai2/prometheus ai2/ceres ai2/jupiter ai2/titan ai2/saturn ai2/neptune"
CKPT_DS="$ATMOS" TAG=rco2rs1-before ./launch-1deg-ace-randco2-evals.sh 2>&1 | tail -4
unset CLUSTERS
for ARM in alt alt-replay; do
  ATMOS_DS="$ATMOS" ARM=$ARM ./launch-1deg-coupled-ft.sh 2>&1 | tail -1
  B=1460; [ "$ARM" = alt-replay ] && B=2190
  (BATCHES_PER_EPOCH=$B EXPERIMENT="troya/1deg-randco2rs1-fmean-$ARM" TAG="rco2rs1-1deg-$ARM" WORK="/home/troya/ace-randco2-figs/watchdog-work-1deg-$ARM" EVAL_LAUNCHER="$SCRIPT_DIR/launch-1deg-ace-randco2-evals.sh" EVAL_JOB_PREFIX=ace1deg-randco2 nohup setsid "$TOOLS/watch_coupled_ft_epochs.sh" >> "/home/troya/ace-randco2-figs/watchdog-1deg-$ARM.log" 2>&1 &)
  log "launched $ARM and its watchdog (batches/epoch $B)"
done
log "done"
