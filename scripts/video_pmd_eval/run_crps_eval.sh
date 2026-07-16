#!/bin/bash
# Runs crps_eval.py as a non-interactive batch job on Beaker (CPU only, weka
# mounted), waits for it to finish, and fetches the results (stdout log +
# two PNG figures) into ./crps_eval_results/.
#
# Unlike an interactive `beaker session create` (no --detach), this submits
# the job, polls for completion, and exits -- no need to stay attached to a
# shell.
#
# Prereqs: beaker CLI installed and authenticated (`beaker account whoami`).
#
# Run:  bash scripts/video_pmd_eval/run_crps_eval.sh
set -e

SESSION_NAME="run-crps-eval-$(date +%s)"
WORKSPACE="ai2/chloeh"
CLUSTER="ai2/phobos"
BUDGET="ai2/atec-climate"
IMAGE="01KS0HKT272A104Y831YXRD949"  # same image the video PMD train/inference jobs use
CPUS=4
MEMORY="32GiB"
RESULT_DIR="./crps_eval_results"
POLL_INTERVAL=15  # seconds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/crps_eval.py"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"

echo "Submitting $SESSION_NAME to $CLUSTER (workspace $WORKSPACE)..."
B64=$(base64 < "$PY_SCRIPT" | tr -d '\n')

CREATE_OUTPUT=$(beaker session create \
    --bare --detach \
    --cluster "$CLUSTER" \
    --budget "$BUDGET" \
    --workspace "$WORKSPACE" \
    --image "beaker://$IMAGE" \
    --mount src=weka,ref=climate-default,dst=/climate-default \
    --cpus "$CPUS" \
    --memory "$MEMORY" \
    --gpus 0 \
    --timeout 30m \
    --name "$SESSION_NAME" \
    --result /results \
    -- bash -c "set -e; echo $B64 | base64 -d > /tmp/crps_eval.py; cd /results; python3 /tmp/crps_eval.py 2>&1 | tee /results/output.log; echo CRPS_EVAL_DONE" 2>&1)

echo "$CREATE_OUTPUT"
SESSION_ID=$(echo "$CREATE_OUTPUT" | grep -oE '01[A-Z0-9]{24}' | head -1)
if [ -z "$SESSION_ID" ]; then
    echo "Error: could not parse session ID from gantry output" >&2
    exit 1
fi
echo "Session ID: $SESSION_ID"

echo "Polling for completion (every ${POLL_INTERVAL}s)..."
while true; do
    STATUS=$(beaker session get "$SESSION_ID" --format json 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    j = d[0] if isinstance(d, list) else d
    print(j.get('status', {}).get('exited', '') and 'done' or 'running')
except Exception:
    print('unknown')
")
    if [ "$STATUS" = "done" ]; then
        break
    fi
    sleep "$POLL_INTERVAL"
done

echo "Session finished. Fetching logs and results..."
beaker session logs "$SESSION_ID" > "$RESULT_DIR/session.log" 2>&1
tail -60 "$RESULT_DIR/session.log"

# Result dataset ID is reported in the create output as "Results ... saved in
# Beaker dataset <id>".
RESULT_DATASET=$(echo "$CREATE_OUTPUT" | grep -oE 'dataset [A-Za-z0-9]+' | awk '{print $2}' | head -1)
if [ -n "$RESULT_DATASET" ]; then
    beaker dataset fetch "$RESULT_DATASET" --output "$RESULT_DIR" 2>&1 | tail -5
    echo "Results saved to $RESULT_DIR/ (output.log, crps_lead_time.png, crps_map.png)"
else
    echo "Warning: could not parse result dataset ID; check $RESULT_DIR/session.log manually" >&2
fi
