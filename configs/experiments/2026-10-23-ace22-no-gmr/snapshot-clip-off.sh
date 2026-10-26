#!/bin/bash

set -e

# Copy of a stage-2 checkpoint with latent global-mean clipping disabled, uploaded as its own
# dataset so run-long36.sh can score it. The clip only acts at inference, and only when its
# envelope (_gm_min/_gm_max) is finite, so setting the envelope to -inf/+inf turns it off and
# leaves the weights unchanged.
#
#   bash snapshot-clip-off.sh <label> <file in training_checkpoints/>
#   bash snapshot-clip-off.sh ep02 best_ckpt.tar         # then: bash run-long36.sh ep02-clipoff best_ckpt.tar
#
# The source is dataset ace22-no-gmr-stage2-<label> if snapshot-ckpt.sh made one, otherwise
# stage 2's result dataset once it has exited 0. Creates dataset ace22-no-gmr-stage2-<label>-clipoff.

LABEL=${1:?usage: snapshot-clip-off.sh <label> <checkpoint file>}
CKPT_FILE=${2:?usage: snapshot-clip-off.sh <label> <checkpoint file>}
STAGE2="ace22-no-gmr-stage2-rs0"
NAME="ace22-no-gmr-stage2-$LABEL-clipoff"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
if beaker dataset get "$BEAKER_USERNAME/$NAME" >/dev/null 2>&1; then
    echo "skip: dataset $NAME exists"
    exit 0
fi
SNAPSHOT="$BEAKER_USERNAME/ace22-no-gmr-stage2-$LABEL"
if beaker dataset get "$SNAPSHOT" >/dev/null 2>&1; then
    SOURCE=$(beaker dataset get "$SNAPSHOT" --format json | jq -r '.[0].id')
    PREFIX="$CKPT_FILE"
else
    STATUS=$(beaker experiment get "$BEAKER_USERNAME/$STAGE2" --format json \
        | python3 -c "import json, sys; j = json.load(sys.stdin)[0]['jobs'][-1]; print(j['status'].get('exitCode', ''), j['result']['beaker'])")
    read -r EXIT_CODE SOURCE <<< "$STATUS"
    if [[ "$EXIT_CODE" != 0 ]]; then
        echo "ERROR: no dataset $SNAPSHOT and $STAGE2 has not exited 0" >&2
        exit 1
    fi
    PREFIX="training_checkpoints/$CKPT_FILE"
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
beaker dataset fetch "$SOURCE" --prefix "$PREFIX" -o "$WORK/in"
mkdir "$WORK/out"
"${PYTHON:-$HOME/miniconda3/envs/fme/bin/python}" - "$WORK/in/$PREFIX" "$WORK/out/$CKPT_FILE" <<'EOF'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = ckpt["stepper"]["step"]["module"]
keys = [k for k in state if k.endswith(("._gm_min", "._gm_max"))]
if sorted(k.rsplit(".", 1)[1] for k in keys) != ["_gm_max", "_gm_min"]:
    sys.exit(f"ERROR: expected one _gm_min and one _gm_max buffer, found {keys}")
for k in keys:
    if not torch.isfinite(state[k]).all():
        sys.exit(f"ERROR: {k} is already non-finite")
    state[k].fill_(float("-inf") if k.endswith("_gm_min") else float("inf"))
torch.save(ckpt, sys.argv[2])
print(f"epoch {ckpt['epoch']}: set {', '.join(keys)} to -inf/+inf")
EOF
beaker dataset create "$WORK/out/$CKPT_FILE" --name "$NAME" --workspace ai2/ace --budget ai2/atec-climate \
    --desc "$CKPT_FILE of $STAGE2 at $LABEL with latent global-mean clipping disabled"
