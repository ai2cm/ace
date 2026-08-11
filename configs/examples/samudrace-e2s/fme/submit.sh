#!/bin/bash
#
# Submit the ace/fme reference-side SamudrACE inference job via gantry.
#
# Runs vanilla fme coupled inference at ace tag v2026.4.0 (single GPU, no
# torchrun) with the SamudrACE CM4 piControl checkpoint, using the config in
# inference-config.yaml next to this script. Gantry clones the source repo
# from GitHub, so this script clones ace at the tag into a temp dir and runs
# gantry from there; the tag exists on the public remote, which is all gantry
# needs. The config file is not part of the ace repo, so it is shipped to the
# job base64-encoded in an env var and decoded in the entrypoint.
#
# SAMUDRACE_ARTIFACTS_DATASET defaults to the uploaded artifact dataset (the
# beaker.org URL below), which mirrors the allenai/SamudrACE-CM4-piControl
# HF repo layout and is mounted at /samudrace-artifacts; the upload script
# that produced it stays with the maintainer. Override the env var to point
# at a different dataset.

set -euo pipefail

# https://beaker.org/orgs/ai2/workspaces/ace/datasets/01KYQZBGSVF220C1QGMHP08GFT
SAMUDRACE_ARTIFACTS_DATASET="${SAMUDRACE_ARTIFACTS_DATASET:-01KYQZBGSVF220C1QGMHP08GFT}"

ACE_TAG="v2026.4.0"
JOB_NAME="samudrace-fme-reference-inference"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_PATH="${SCRIPT_DIR}/inference-config.yaml"

# Best-effort local validation (requires fme==2026.4.0 in the current env).
if python -c "import fme.coupled.validate_config" 2>/dev/null; then
    python -m fme.coupled.validate_config "$CONFIG_PATH" --config_type inference
else
    echo "WARNING: fme not importable locally; skipping config validation." >&2
fi

CONFIG_B64=$(base64 -w0 "$CONFIG_PATH")

CLONE_DIR=$(mktemp -d)
trap 'rm -rf "$CLONE_DIR"' EXIT
git clone --depth 1 --branch "$ACE_TAG" https://github.com/ai2cm/ace.git "$CLONE_DIR"
cd "$CLONE_DIR"

# No --beaker-image: gantry builds the environment from the repo's pyproject.
#
# torch is pinned to a CUDA 12.8 build from the PyTorch index BEFORE the
# project install. fme only asks for `torch>=2.4.0`, which now resolves on
# PyPI to torch 2.13, a CUDA 13 build (nvidia-*-cu13 wheels). The ai2 A100
# nodes run driver 570.124.06 (CUDA 12.8); CUDA 13 needs r580+, so
# torch.cuda.is_available() would be False and the job would run on CPU (this
# is what happened to the earth2studio-side job, ../e2s/). The subsequent
# `pip install .` leaves this torch in place. The entrypoint then asserts a
# GPU is visible before starting inference, so a bad wheel fails fast instead
# of silently degrading to CPU.
# Outputs go to /results, the job's Beaker result dataset (gantry default),
# matching experiment_dir in the config.
gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "SamudrACE CM4 piControl reference inference (fme ${ACE_TAG}, 24 coupled cycles, scenario 0311)" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 2h \
    --cluster ai2/saturn-cirrascale \
    --gpus 1 \
    --shared-memory 50GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && pip install . && python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available())'" \
    --dataset "${SAMUDRACE_ARTIFACTS_DATASET}:/samudrace-artifacts" \
    --env INFERENCE_CONFIG_B64="$CONFIG_B64" \
    -- bash -c 'python -c "
import os, sys, torch
if torch.cuda.is_available():
    sys.exit(0)
if os.environ.get(\"SAMUDRACE_ALLOW_CPU\") == \"1\":
    print(\"WARNING: no CUDA device; running on CPU because SAMUDRACE_ALLOW_CPU=1\")
    sys.exit(0)
sys.exit(
    f\"REFUSING TO RUN: torch.cuda.is_available() is False (torch {torch.__version__}, \"
    f\"built for CUDA {torch.version.cuda}). This job requests a GPU; running on \"
    \"CPU would waste hours. Set SAMUDRACE_ALLOW_CPU=1 to override.\"
)
" && echo "$INFERENCE_CONFIG_B64" | base64 -d > /tmp/inference-config.yaml && python -m fme.coupled.inference /tmp/inference-config.yaml'
