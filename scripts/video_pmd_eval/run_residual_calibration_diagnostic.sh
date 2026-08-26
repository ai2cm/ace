#!/bin/bash
# Runs residual_calibration_diagnostic.py as a non-interactive batch job on
# Beaker (CPU only, weka mounted), waits for it to finish, and fetches the
# results (stdout log + JSON) into ./residual_calibration_results/.
#
# Same launch pattern as run_d_block_characterization.sh (see
# run_crps_eval.sh's header for the full pattern) -- submits, polls for
# completion, exits.
#
# Prereqs: beaker CLI installed and authenticated (`beaker account whoami`).
#
# Run:  bash scripts/video_pmd_eval/run_residual_calibration_diagnostic.sh
set -e

SESSION_NAME="run-residual-calibration-$(date +%s)"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/neptune"
PRIORITY="urgent"
BUDGET="ai2/atec-climate"
IMAGE="01KS0HKT272A104Y831YXRD949"  # same image the video PMD train/inference jobs use
CPUS=8
MEMORY="64GiB"  # global 25km fields are ~16x a 1-degree grid's pixel count
RESULT_DIR="./residual_calibration_results"
POLL_INTERVAL=15  # seconds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/residual_calibration_diagnostic.py"

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
    --priority "$PRIORITY" \
    --budget "$BUDGET" \
    --workspace "$WORKSPACE" \
    --image "beaker://$IMAGE" \
    --mount "src=weka,ref=climate-default,dst=/climate-default" \
    --cpus "$CPUS" \
    --memory "$MEMORY" \
    --gpus 0 \
    --timeout 3h \
    --name "$SESSION_NAME" \
    --result /results \
    -- bash -c "set -e; echo $B64 | base64 -d > /tmp/residual_calibration_diagnostic.py; cd /results; python3 -u /tmp/residual_calibration_diagnostic.py 2>&1 | tee /results/output.log; echo RESIDUAL_CALIBRATION_DONE" 2>&1)

echo "$CREATE_OUTPUT"
SESSION_ID=$(echo "$CREATE_OUTPUT" | grep -oE '01[A-Z0-9]{24}' | head -1)
if [ -z "$SESSION_ID" ]; then
    echo "Error: could not parse session ID from beaker output" >&2
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
tail -100 "$RESULT_DIR/session.log"

RESULT_DATASET=$(echo "$CREATE_OUTPUT" | grep -oE 'dataset [A-Za-z0-9]+' | awk '{print $2}' | head -1)
if [ -n "$RESULT_DATASET" ]; then
    beaker dataset fetch "$RESULT_DATASET" --output "$RESULT_DIR" 2>&1 | tail -5
    echo "Results saved to $RESULT_DIR/ (output.log, residual_calibration_results.json)"
else
    echo "Warning: could not parse result dataset ID; check $RESULT_DIR/session.log manually" >&2
fi
