#!/bin/bash

set -e

# Copy one checkpoint out of the running stage 2 into its own dataset, so run-long36.sh can
# score it before stage 2 finishes. Beaker will not mount a running job's result dataset, but
# its files can be fetched.
#
#   bash snapshot-ckpt.sh <label> <file in training_checkpoints/> <expected epoch>
#   bash snapshot-ckpt.sh ep01 best_ckpt.tar 1
#
# Uploads only if the checkpoint's stored epoch is the expected one, so a best_ckpt.tar that
# has already moved on to a later epoch is caught. Creates dataset ace22-no-gmr-stage2-<label>.

LABEL=${1:?usage: snapshot-ckpt.sh <label> <checkpoint file>}
CKPT_FILE=${2:?usage: snapshot-ckpt.sh <label> <checkpoint file> <expected epoch>}
EXPECTED_EPOCH=${3:?usage: snapshot-ckpt.sh <label> <checkpoint file> <expected epoch>}
STAGE2="ace22-no-gmr-stage2-rs0"
NAME="ace22-no-gmr-stage2-$LABEL"

BEAKER_USERNAME=$(beaker account whoami --format=json | jq -r '.[0].name')
if beaker dataset get "$BEAKER_USERNAME/$NAME" >/dev/null 2>&1; then
    echo "skip: dataset $NAME exists"
    exit 0
fi
RESULT=$(beaker experiment get "$BEAKER_USERNAME/$STAGE2" --format json \
    | python3 -c "import json, sys; print(json.load(sys.stdin)[0]['jobs'][-1]['result']['beaker'])")
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
beaker dataset fetch "$RESULT" --prefix "training_checkpoints/$CKPT_FILE" -o "$WORK"
CKPT="$WORK/training_checkpoints/$CKPT_FILE"
EPOCH=$("${PYTHON:-$HOME/miniconda3/envs/fme/bin/python}" -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$CKPT")
if [[ "$EPOCH" != "$EXPECTED_EPOCH" ]]; then
    echo "ERROR: $CKPT_FILE is from epoch $EPOCH, not $EXPECTED_EPOCH; not uploaded" >&2
    exit 1
fi
echo "$CKPT_FILE is from epoch $EPOCH; uploading as $NAME"
beaker dataset create "$CKPT" --name "$NAME" --workspace ai2/ace --budget ai2/atec-climate \
    --desc "Stage-2 $CKPT_FILE of $STAGE2 at $LABEL, copied while training ran"
