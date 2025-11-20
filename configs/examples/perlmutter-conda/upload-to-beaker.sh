#!/bin/bash

SLURM_JOB_ID=$1

PATH_TO_UPLOAD=$SCRATCH/fme-output/$SLURM_JOB_ID

# Try to get wandb link in order to provide more useful linking.
# This assumes logs still remain where the batch jobs are configured to save them.
MAYBE_LOG_PATH="joblogs/$SLURM_JOB_ID.out"
if [ -f "$MAYBE_LOG_PATH" ]; then
    echo "Trying to get wandb link from $MAYBE_LOG_PATH"
    WANDB_URL=$(head -n 200 $MAYBE_LOG_PATH | grep 'https://wandb.ai.*/runs/' | cut -c 22- )
    echo $WANDB_URL
else
    WANDB_URL="unknown"
fi

NAME="perlmutter-upload-$SLURM_JOB_ID"
DESCRIPTION="Uploaded output from perlmutter job $SLURM_JOB_ID. \
You may be able to find the corresponding wandb run here: $WANDB_URL. \
The beaker dataset was created from $PATH_TO_UPLOAD."

echo "Uploading $PATH_TO_UPLOAD to beaker dataset named $NAME"

beaker dataset create \
    --name "$NAME" \
    --desc "$DESCRIPTION" \
    $PATH_TO_UPLOAD
