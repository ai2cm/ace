#!/bin/bash
# Watchdog (Troy 2026-10-15): after EVERY epoch of a coupled fine-tune, lift the atmosphere (EMA weights) out of that
# epoch's checkpoint and launch the ACE-only random-CO2 evaluations on it, so the per-epoch drift of ACE's CO2 response
# can be tracked (not just the selected checkpoint). Polls the Beaker experiment: for each segment's log line
# "Saving checkpoints for epoch N" (written after the epoch's inline inference) it fetches that segment's
# training_checkpoints/ckpt.tar, checks it is the end-of-epoch save (num_batches_seen == N x batches/epoch; the trainer also overwrites
# ckpt.tar mid-epoch every 1000 batches, ~1 h later, so the poll must be faster than that),
# lifts with --ema, uploads ace4deg-atmos-lifted-<TAG>-ep<N> and runs launch-4deg-ace-randco2-evals.sh with
# TAG=<TAG>-ep<N>. Epochs done are recorded in $STATE so the loop is restartable. Exits when the experiment has
# exited and no unprocessed epoch remains, or after MAX_EPOCHS.
#   EXPERIMENT=troya/samudra-ppad4deg-ff-ohc-ppad-coupled-ft-jointens-rco2rs2 TAG=rco2rs2-after-jointens \
#     nohup setsid tools/watch_coupled_ft_epochs.sh > /home/troya/ace-randco2-figs/watchdog.log 2>&1 &
set -uo pipefail
EXPERIMENT="${EXPERIMENT:?beaker experiment name or id}"
TAG="${TAG:?tag prefix, e.g. rco2rs2-after-jointens}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
POLL="${POLL:-300}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-1460}"   # 160 yr x 73 ocean steps / batch 8
WORK="${WORK:-/home/troya/ace-randco2-figs/watchdog-work}"
STATE="${STATE:-/home/troya/ace-randco2-figs/watchdog-${TAG}.done}"
export CLUSTERS="${CLUSTERS:-ai2/rhea ai2/prometheus ai2/ceres ai2/jupiter ai2/titan ai2/saturn ai2/neptune}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/launch-4deg-ace-randco2-evals.sh}"   # the ACE-only evaluator launcher to run per epoch
REPO_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$WORK"; touch "$STATE"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "watchdog start: $EXPERIMENT tag=$TAG max_epochs=$MAX_EPOCHS poll=${POLL}s"
while true; do
  # segments: job id, results dataset, exited?
  segs=$(beaker experiment get "$EXPERIMENT" --format json 2>/dev/null | python3 -c "
import json,sys
e=json.load(sys.stdin)[0]
for j in e['jobs']:
    s=j['status']
    # 'finished' only when the segment exited cleanly (code 0); a preempted segment (non-zero exit) is
    # resumed by Beaker a little later, so it must not count as the experiment having ended
    state='finished' if s.get('exited') and s.get('exitCode')==0 else ('exited' if s.get('exited') else 'running')
    print(j['id'], j.get('result',{}).get('beaker'), state)") || { log "beaker experiment get failed"; sleep "$POLL"; continue; }
  if [ -z "$segs" ]; then log "no segments returned (transient?); retrying"; sleep "$POLL"; continue; fi
  all_exited=1
  while read -r JOB DS STATE_J; do
    [ "$STATE_J" = finished ] || all_exited=0
    # epochs saved in this segment
    for N in $(beaker job logs "$JOB" 2>/dev/null | grep -o "Saving checkpoints for epoch [0-9]*" | awk '{print $5}' | sort -un); do
      grep -qx "$N" "$STATE" && continue
      grep -qsx "$N" "$STATE.lost" && continue
      [ "$N" -ge 1 ] || continue
      log "epoch $N saved in segment $JOB (dataset $DS): fetching ckpt.tar"
      D="$WORK/ep$N"; rm -rf "$D"; mkdir -p "$D/src" "$D/upload/training_checkpoints"
      if ! beaker dataset fetch "$DS" --prefix training_checkpoints/ckpt.tar -o "$D/src" >"$D/fetch.log" 2>&1; then
        log "epoch $N: fetch failed ($(tail -1 "$D/fetch.log")); will retry"; continue
      fi
      info=$(PYTHONPATH="$REPO_ROOT" python "$SCRIPT_DIR/extract_atmosphere_from_coupled_ckpt.py" \
             "$D/src/training_checkpoints/ckpt.tar" "$D/upload/training_checkpoints/best_inference_ckpt.tar" --ema 2>&1 | grep "^wrote")
      log "epoch $N: $info"
      # the restart checkpoint is written at the end of train_one_epoch, BEFORE the epoch counter increments and before
      # the inline evaluation: the end-of-epoch-N file reads "epoch N-1, batches N*B (+B this epoch)"
      got=$(echo "$info" | sed -n 's/.*batches \([0-9]*\) .*/\1/p')
      want=$((N * BATCHES_PER_EPOCH))
      if [ "${got:-0}" -ne "$want" ]; then
        if [ "${got:-0}" -gt "$want" ]; then
          log "epoch $N: ckpt.tar in $DS is already past epoch $N ($got > $want batches); checkpoint lost, not retrying"
          echo "$N" >> "$STATE.lost"
        else
          log "epoch $N: ckpt.tar has $got batches, end of epoch $N is $want; not uploaded yet, will retry"
        fi
        rm -rf "$D"; continue
      fi
      NAME="ace4deg-atmos-lifted-${TAG}-ep${N}"
      if ! beaker dataset create --workspace ai2/ace --name "$NAME" \
           --desc "EMA atmosphere lifted from $EXPERIMENT ckpt.tar at the end of epoch $N (segment $DS)" "$D/upload" >"$D/upload.log" 2>&1; then
        log "epoch $N: dataset upload failed: $(tail -2 "$D/upload.log" | tr '\n' ' ')"; continue
      fi
      LIFTED=$(beaker dataset get "$(beaker account whoami --format=json | jq -r '.[0].name')/$NAME" --format json | jq -r '.[0].id')
      log "epoch $N: lifted atmosphere dataset $LIFTED; launching random-CO2 evals"
      (cd "$(dirname "$EVAL_LAUNCHER")" && CKPT_DS="$LIFTED" TAG="${TAG}-ep${N}" "./$(basename "$EVAL_LAUNCHER")" 2>&1 | sed "s/^/  /")
      echo "$N" >> "$STATE"
      rm -rf "$D/src"
    done
  done <<< "$segs"
  # refresh the analysis once an epoch's three evaluations have all exited
  for N in $(cat "$STATE"); do
    grep -qx "$N" "$STATE.refreshed" 2>/dev/null && continue
    n_ok=0
    for X in 1 2 4; do
      beaker experiment get "troya/${EVAL_JOB_PREFIX:-ace4deg-randco2}-${X}xCO2-ic3-${TAG}-ep${N}" --format json 2>/dev/null \
        | python3 -c "import json,sys; s=json.load(sys.stdin)[0]['jobs'][-1]['status']; sys.exit(0 if s.get('exited') and s.get('exitCode')==0 else 1)" \
        && n_ok=$((n_ok+1))
    done
    if [ "$n_ok" -eq 3 ]; then
      log "epoch $N: all three evaluations finished; refreshing the analysis"
      "$SCRIPT_DIR/refresh_randco2_analysis.sh" 2>&1 | sed "s/^/  /" && echo "$N" >> "$STATE.refreshed"
    fi
  done
  n_done=$(( $(wc -l < "$STATE") + $(cat "$STATE.lost" 2>/dev/null | wc -l) ))
  if [ "$n_done" -ge "$MAX_EPOCHS" ]; then log "all $MAX_EPOCHS epochs processed; exiting"; exit 0; fi
  if [ "$all_exited" = 1 ]; then log "experiment finished (last segment exit 0) with $n_done epochs processed; exiting"; exit 0; fi
  sleep "$POLL"
done
