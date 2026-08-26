#!/bin/bash
# Runs endpoint_vs_interior_diagnostic.py as a non-interactive batch job on
# Beaker (CPU only, weka mounted), waits for it to finish, and fetches the
# results (stdout log + CSV) into ./endpoint_vs_interior_results_<model>/.
#
# Same launch pattern as run_crps_eval.sh (see that script's header) --
# submits, polls for completion, exits.
#
# Prereqs: beaker CLI installed and authenticated (`beaker account whoami`).
#
# Run:  bash scripts/video_pmd_eval/run_endpoint_vs_interior_diagnostic.sh [model]
#   model: one of endpoint_vs_interior_diagnostic.py's PATCHED_MODELS keys
#     (default: st-singlestage-coarse-endpoints-flat).
set -e

MODEL="${1:-st-singlestage-coarse-endpoints-flat}"
SESSION_NAME="run-endpoint-vs-interior-${MODEL}-$(date +%s)"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/neptune"
PRIORITY="urgent"
BUDGET="ai2/atec-climate"
IMAGE="01KS0HKT272A104Y831YXRD949"  # same image the video PMD train/inference jobs use
CPUS=4
MEMORY="32GiB"
RESULT_DIR="./endpoint_vs_interior_results_${MODEL}"
POLL_INTERVAL=15  # seconds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/endpoint_vs_interior_diagnostic.py"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"

echo "Submitting $SESSION_NAME (model=$MODEL) to $CLUSTER (workspace $WORKSPACE)..."
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
    -- bash -c "set -e; echo $B64 | base64 -d > /tmp/endpoint_vs_interior_diagnostic.py; cd /results; python3 -u /tmp/endpoint_vs_interior_diagnostic.py --model $MODEL --outdir /results 2>&1 | tee /results/output.log; echo DIAGNOSTIC_DONE" 2>&1)

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
tail -80 "$RESULT_DIR/session.log"

RESULT_DATASET=$(echo "$CREATE_OUTPUT" | grep -oE 'dataset [A-Za-z0-9]+' | awk '{print $2}' | head -1)
if [ -n "$RESULT_DATASET" ]; then
    beaker dataset fetch "$RESULT_DATASET" --output "$RESULT_DIR" 2>&1 | tail -5
    echo "Results saved to $RESULT_DIR/ (output.log, endpoint_vs_interior_${MODEL}.csv)"
else
    echo "Warning: could not parse result dataset ID; check $RESULT_DIR/session.log manually" >&2
fi
