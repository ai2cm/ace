#!/bin/bash
#
# Submit the earth2studio-side SamudrACE inference job via gantry.
#
# Runs earth2studio's `run.deterministic` through the SamudrACE prognostic
# wrapper with the SamudrACE CM4 piControl checkpoint (single GPU), using the
# payload in run_inference.py next to this script. Gantry clones the source
# repo from GitHub, so this script clones the earth2studio fork branch into a
# temp dir and runs gantry from there; the branch must exist on the remote.
# The payload is not part of the repo, so it is shipped to the job
# base64-encoded in an env var and decoded in the entrypoint.
#
# SAMUDRACE_ARTIFACTS_DATASET defaults to the uploaded artifact dataset (the
# beaker.org URL below), which mirrors the allenai/SamudrACE-CM4-piControl
# HF repo layout and is mounted at /samudrace-artifacts; the upload script
# that produced it stays with the maintainer. Override the env var to point
# at a different dataset.

set -euo pipefail

# https://beaker.org/orgs/ai2/workspaces/ace/datasets/01KYQZBGSVF220C1QGMHP08GFT
SAMUDRACE_ARTIFACTS_DATASET="${SAMUDRACE_ARTIFACTS_DATASET:-01KYQZBGSVF220C1QGMHP08GFT}"

E2S_REPO="https://github.com/jpdunc23/earth2studio.git"
E2S_BRANCH="feature/samudrace-predict-paired"
JOB_NAME="samudrace-e2s-inference"
N_COUPLED_CYCLES="${SAMUDRACE_N_COUPLED_CYCLES:-24}"
SCENARIO="${SAMUDRACE_SCENARIO:-0311}"
IC_TIME="${SAMUDRACE_IC_TIME:-0311-01-01T00:00:00}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAYLOAD_PATH="${SCRIPT_DIR}/run_inference.py"

# Cheap local check that the payload at least parses.
python -m py_compile "$PAYLOAD_PATH"

PAYLOAD_B64=$(base64 -w0 "$PAYLOAD_PATH")

CLONE_DIR=$(mktemp -d)
trap 'rm -rf "$CLONE_DIR"' EXIT
git clone --depth 1 --branch "$E2S_BRANCH" "$E2S_REPO" "$CLONE_DIR"
cd "$CLONE_DIR"

# No --beaker-image: gantry builds the environment from the cloned repo's
# pyproject. The `samudrace` extra pulls fme==2026.4.0 plus pandas/scipy, all
# from PyPI, so plain pip resolves it without uv's [tool.uv.sources] pins.
#
# torch is pinned to a CUDA 12.8 build from the PyTorch index BEFORE the
# project install. earth2studio only asks for `torch>=2.5.0`, which now
# resolves on PyPI to torch 2.13, a CUDA 13 build (nvidia-*-cu13 wheels). The
# ai2 A100 nodes run driver 570.124.06 (CUDA 12.8); CUDA 13 needs r580+, so
# torch.cuda.is_available() is False there and the job silently ran on CPU.
# The subsequent `pip install '.[samudrace]'` leaves this torch in place
# (2.9.1 satisfies both `torch>=2.5.0` and fme's `torch>=2.4.0`).
# Outputs go to /results, the job's Beaker result dataset (gantry default).
# HF_HUB_OFFLINE guards against any accidental HuggingFace fetch: every model
# input comes from the mounted artifact dataset.
gantry run \
    --name "$JOB_NAME" \
    --task-name "$JOB_NAME" \
    --description "SamudrACE CM4 piControl earth2studio inference (${N_COUPLED_CYCLES} coupled cycles, scenario ${SCENARIO})" \
    --workspace ai2/ace \
    --priority high \
    --min-runtime 2h \
    --cluster ai2/saturn-cirrascale \
    --gpus 1 \
    --shared-memory 50GiB \
    --budget ai2/atec-climate \
    --system-python \
    --install "pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && pip install '.[samudrace]' && python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available())'" \
    --dataset "${SAMUDRACE_ARTIFACTS_DATASET}:/samudrace-artifacts" \
    --env SAMUDRACE_ARTIFACTS=/samudrace-artifacts \
    --env SAMUDRACE_OUTPUT_DIR=/results \
    --env SAMUDRACE_N_COUPLED_CYCLES="$N_COUPLED_CYCLES" \
    --env SAMUDRACE_SCENARIO="$SCENARIO" \
    --env SAMUDRACE_IC_TIME="$IC_TIME" \
    --env HF_HUB_OFFLINE=1 \
    --env RUN_INFERENCE_B64="$PAYLOAD_B64" \
    -- bash -c 'echo "$RUN_INFERENCE_B64" | base64 -d > /tmp/run_inference.py && python /tmp/run_inference.py'
