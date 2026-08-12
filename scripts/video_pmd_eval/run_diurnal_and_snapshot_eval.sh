#!/bin/bash
# Runs diurnal_cycle_eval.py + global_snapshot_eval.py as one non-interactive
# Beaker batch job (CPU, weka mounted) for st-singlestage-flat, matching the
# st-flat/st-ou methodology in crps_eval_results_stage2_st-flat-st-ou/.
#
# Run:  bash scripts/video_pmd_eval/run_diurnal_and_snapshot_eval.sh
set -e

MODEL="${1:-st-singlestage-flat}"
JOB_NAME="diurnal-snapshot-${MODEL}-$(date +%s)"
WORKSPACE="ai2/climate-titan"
CLUSTER="ai2/saturn"
CPUS=4
MEMORY="96GiB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DEPS_ONLY_IMAGE="$(cat latest_deps_only_image.txt)"

DIURNAL_B64=$(base64 < scripts/video_pmd_eval/diurnal_cycle_eval.py | tr -d '\n')
SNAPSHOT_B64=$(base64 < scripts/video_pmd_eval/global_snapshot_eval.py | tr -d '\n')

set +e
CREATE_OUTPUT=$(beaker session create \
    --bare --detach \
    --cluster "$CLUSTER" \
    --priority urgent \
    --budget ai2/atec-climate \
    --workspace "$WORKSPACE" \
    --image "beaker://$DEPS_ONLY_IMAGE" \
    --mount src=weka,ref=climate-default,dst=/climate-default \
    --cpus "$CPUS" \
    --memory "$MEMORY" \
    --gpus 0 \
    --timeout 1h \
    --name "$JOB_NAME" \
    --result /results \
    -- bash -c "
set -e
echo $DIURNAL_B64 | base64 -d > /tmp/diurnal_cycle_eval.py
echo $SNAPSHOT_B64 | base64 -d > /tmp/global_snapshot_eval.py
cd /results
python3 /tmp/diurnal_cycle_eval.py --model $MODEL --outdir /results
python3 /tmp/global_snapshot_eval.py --model $MODEL --outdir /results
echo DIURNAL_SNAPSHOT_DONE
" 2>&1)
set -e

echo "$CREATE_OUTPUT"
SESSION_ID=$(echo "$CREATE_OUTPUT" | grep -oE '01[A-Z0-9]{24}' | head -1)
if [ -z "$SESSION_ID" ]; then
    echo "Error: could not parse session ID from beaker output" >&2
    exit 1
fi
echo "Session ID: $SESSION_ID"
echo "Follow logs with: beaker session logs -f $SESSION_ID"
