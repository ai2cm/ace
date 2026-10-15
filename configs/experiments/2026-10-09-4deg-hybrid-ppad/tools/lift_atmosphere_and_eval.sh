#!/bin/bash
# "ACE after": fetch a coupled run's checkpoint, lift the atmosphere out of it, upload it as a Beaker dataset and launch
# the ACE-only random-CO2 evaluations (eval-ace/) on it. Run from a machine with beaker + the fme env (torch + fme).
#   RESULTS_DS=<coupled run's (finalized) results dataset> TAG=rco2rs2-after-jointens-ep1 tools/lift_atmosphere_and_eval.sh
#   CKPT_FILE=training_checkpoints/best_inference_ckpt.tar (default) or ckpt.tar; CLUSTERS/PRIORITY pass through.
set -euo pipefail
RESULTS_DS="${RESULTS_DS:?coupled run results dataset}"
TAG="${TAG:?label, e.g. rco2rs2-after-jointens-ep1}"
CKPT_FILE="${CKPT_FILE:-training_checkpoints/best_inference_ckpt.tar}"
WORK="${WORK:-/tmp/lift-$TAG}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$WORK/src" "$WORK/upload/training_checkpoints"
[ -f "$WORK/src/$CKPT_FILE" ] || beaker dataset fetch "$RESULTS_DS" --prefix "$CKPT_FILE" -o "$WORK/src"
PYTHONPATH="$REPO_ROOT" python "$SCRIPT_DIR/extract_atmosphere_from_coupled_ckpt.py" \
  "$WORK/src/$CKPT_FILE" "$WORK/upload/training_checkpoints/best_inference_ckpt.tar"
NAME="ace4deg-atmos-lifted-${TAG}"
beaker dataset create --workspace ai2/ace --name "$NAME" \
  --desc "atmosphere lifted (extract_atmosphere_from_coupled_ckpt.py) from ${RESULTS_DS}:${CKPT_FILE}" "$WORK/upload"
DS=$(beaker dataset get "$(beaker account whoami --format=json | jq -r '.[0].name')/$NAME" --format json | jq -r '.[0].id')
echo "lifted atmosphere dataset: $DS"
cd "$SCRIPT_DIR/.."
CKPT_DS="$DS" TAG="$TAG" ./launch-4deg-ace-randco2-evals.sh
