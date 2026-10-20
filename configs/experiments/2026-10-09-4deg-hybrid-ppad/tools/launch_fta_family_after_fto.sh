#!/bin/bash
# Waits for the rs2 FTO run (-ens-rco2rs2) to finish, lifts its selected ocean (best_inference_ckpt) into a standalone
# ocean checkpoint, uploads it, then launches the FTA-family arms against it and starts a per-epoch watchdog for each.
# Troy 2026-10-20 adaptation experiments. Run detached: nohup setsid tools/launch_fta_family_after_fto.sh > ... &
set -uo pipefail
EXPERIMENT=troya/samudra-ppad4deg-ff-ohc-ppad-coupled-ft-ens-rco2rs2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
WORK=/home/troya/ace-randco2-figs/lift-fto-rco2rs2-ocean
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for $EXPERIMENT to finish"
while true; do
  st=$(beaker experiment get "$EXPERIMENT" --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]; j=e['jobs'][-1]; s=j['status']
print(('done '+j['result']['beaker']) if s.get('exited') and s.get('exitCode')==0 else ('failed' if s.get('exited') and s.get('exitCode') not in (None,0) and len(e['jobs'])>40 else 'running'))" 2>/dev/null)
  case "$st" in done*) DS=${st#done }; break;; esac
  sleep 600
done
log "FTO finished, results dataset $DS; lifting the selected ocean"
mkdir -p "$WORK/src" "$WORK/upload/training_checkpoints"
beaker dataset fetch "$DS" --prefix training_checkpoints/best_inference_ckpt.tar -o "$WORK/src" > "$WORK/fetch.log" 2>&1
PYTHONPATH="$REPO_ROOT" python "$SCRIPT_DIR/extract_ocean_from_coupled_ckpt.py" "$WORK/src/training_checkpoints/best_inference_ckpt.tar" "$WORK/upload/training_checkpoints/best_inference_ckpt.tar" 2>&1 | grep -E "wrote|Error"
beaker dataset create --workspace ai2/ace --name samudra-ppad4deg-ff-ohc-ppad-fto-rco2rs2-selected-ocean --desc "ocean lifted from $EXPERIMENT best_inference_ckpt ($DS)" "$WORK/upload" > "$WORK/upload.log" 2>&1
OCEAN=$(beaker dataset get troya/samudra-ppad4deg-ff-ohc-ppad-fto-rco2rs2-selected-ocean --format json | jq -r '.[0].id')
log "lifted ocean dataset $OCEAN; launching FTA-family arms"
cd "$SCRIPT_DIR/.."
for V in lr1e-5 lr7e-6 lr1e-5-1yr lr1e-5-10yr encdec lastblock firstblock; do
  OCEAN_DS_OVERRIDE="$OCEAN" ATMOS_DS=01M4Z3CQXD8BF13S4X6462XVNH ARMS=ff-ohc-ppad VARIANT="-fta-rco2rs2-$V" ./launch-4deg-ppad-coupled-ft.sh 2>&1 | tail -1
  case "$V" in lr1e-5-1yr) B=18; E=1;; lr1e-5-10yr) B=183; E=1;; lr1e-5|lr7e-6) B=1460; E=1;; *) B=1460; E=3;; esac
  (BATCHES_PER_EPOCH=$B MAX_EPOCHS=$E EXPERIMENT="troya/samudra-ppad4deg-ff-ohc-ppad-coupled-ft-fta-rco2rs2-$V" TAG="rco2rs2-fta-$V" WORK="/home/troya/ace-randco2-figs/watchdog-work-fta-$V" nohup setsid "$SCRIPT_DIR/watch_coupled_ft_epochs.sh" >> "/home/troya/ace-randco2-figs/watchdog-fta-$V.log" 2>&1 &)
  log "watchdog started for $V (batches/epoch $B, epochs $E)"
done
log "all FTA-family arms launched"
