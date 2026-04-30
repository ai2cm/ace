#!/bin/bash

# Download a CoupledStepper training results Beaker dataset, extract an
# individual component Stepper, and create a new Beaker dataset with the
# component checkpoint which can then be used for uncoupled inference.
#
# Usage:
#   create_decoupled_checkpoint_beaker.sh [OPTIONS]
#
# Required Options:
#   --id          CoupledStepper training Beaker dataset ID.
#   --prefix      Path in the Beaker dataset to the checkpoint.
#   --component   Component to extract: 'ocean' or 'atmosphere'.
#
# Example:
#   bash create_decoupled_checkpoint_beaker.sh  \
#       --id 01KCA5QX2K7FQ2NXDE4YTY3ZE6 \
#       --prefix training_checkpoints/best_inference_ckpt.tar \
#       --component atmosphere

DESCRIPTION="$0 $*"
echo "$DESCRIPTION"

# Parse command line arguments
while [[ "$#" -gt 0 ]]
do case $1 in
    --id) DATASET_ID="$2"
    shift;;
    --prefix) PREFIX="$2"
    shift;;
    --component) COMPONENT="$2"
    shift;;
    *) echo "Unknown parameter passed: $1"
    exit 1;;
esac
shift
done

UUID=$(uuidgen)
OUTPUT_DIR=/tmp
CKPT_PATH="$OUTPUT_DIR/$UUID/$PREFIX"

mkdir -p "$(dirname "$CKPT_PATH")"

set -e

# Fetch checkpoint
beaker dataset fetch "$DATASET_ID" --output "$OUTPUT_DIR/$DATASET_ID" --prefix "$PREFIX"

python -u create_decoupled_checkpoint.py \
    --component "$COMPONENT" \
    --input_path "$OUTPUT_DIR/$DATASET_ID/$PREFIX" \
    --output_path "$CKPT_PATH"

beaker dataset create -b ai2/climate "$OUTPUT_DIR/$UUID" --desc "$DESCRIPTION"

exit 0
