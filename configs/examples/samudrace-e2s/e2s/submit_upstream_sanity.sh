#!/bin/bash
#
# Test the upstream-facing SamudrACE sanity-check script on a GPU.
#
# Clones the earth2studio fork branch, installs it, ships the script in an env
# var, and runs it. Everything the script needs (checkpoint, ICs, forcing) is
# downloaded from HuggingFace at run time, so no artifact dataset is mounted
# and HF_HUB_OFFLINE is deliberately NOT set.

set -euo pipefail

E2S_REPO="https://github.com/elynnwu/earth2studio.git"
E2S_BRANCH="feature/samudrace-predict-paired"
JOB_NAME="samudrace-e2s-sanity-check"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAYLOAD_PATH="${SCRIPT_DIR}/samudrace_sanity_check.py"

python -m py_compile "$PAYLOAD_PATH"
PAYLOAD_B64=$(base64 -w0 "$PAYLOAD_PATH" 2>/dev/null || base64 -i "$PAYLOAD_PATH" | tr -d '\n')

CLONE_DIR=$(mktemp -d)
trap 'rm -rf "$CLONE_DIR"' EXIT
git clone --depth 1 --branch "$E2S_BRANCH" "$E2S_REPO" "$CLONE_DIR"
cd "$CLONE_DIR"

gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "SamudrACE upstream sanity-check script (HF checkpoint, 24 coupled cycles, contour figure)" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 2h \
    --cluster ai2/saturn-cirrascale \
    --gpus 1 \
    --shared-memory 50GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && pip install '.[samudrace]' && python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available())'" \
    --env SANITY_B64="$PAYLOAD_B64" \
    -- bash -c 'echo "$SANITY_B64" | base64 -d > /tmp/samudrace_sanity_check.py && cd /results && python /tmp/samudrace_sanity_check.py'
