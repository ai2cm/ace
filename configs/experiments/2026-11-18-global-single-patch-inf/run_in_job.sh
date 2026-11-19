#!/usr/bin/env bash
# Entry point inside a Beaker job (submitted by submit.sh).
#
#   run_in_job.sh CONFIG.yaml [INPUT_PATH ...]
#
# Writes outputs to $AGENT_OUTPUT_DIR (set by the Beaker proxy), so the config's
# experiment_dir placeholder is replaced there. The trailing input paths are only
# listed so the proxy mounts their top-level weka folders.
set -euo pipefail
CONFIG=$1
echo "AGENT_OUTPUT_DIR=${AGENT_OUTPUT_DIR:?}"
mkdir -p "$AGENT_OUTPUT_DIR"

RUN_CONFIG=/tmp/$(basename "$CONFIG")
sed "s|^experiment_dir: .*|experiment_dir: ${AGENT_OUTPUT_DIR}|" "$CONFIG" > "$RUN_CONFIG"
cp "$RUN_CONFIG" "$AGENT_OUTPUT_DIR/config.yaml"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$AGENT_OUTPUT_DIR/gpu.txt"

# Restart-safe: a resumed or preempted attempt overwrites the same event files.
python -m fme.downscaling.evaluator "$RUN_CONFIG"
ls -la "$AGENT_OUTPUT_DIR"
