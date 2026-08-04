#!/bin/bash

# numeric PBS job id (e.g. 1234567 from 1234567.polaris-pbs-01...)
JOBID=$1

FME_SCRATCH=${FME_SCRATCH:-/eagle/E3SMinput/elynnwu/scratch}
PATH_TO_UPLOAD=$FME_SCRATCH/fme-output/$JOBID

# Try to get wandb link in order to provide more useful linking.
# Best-effort: the PBS job writes its log to joblogs/<jobname>.o<jobid>.
MAYBE_LOG_PATH=$(ls joblogs/*.o${JOBID} 2>/dev/null | head -n 1)
if [ -n "$MAYBE_LOG_PATH" ] && [ -f "$MAYBE_LOG_PATH" ]; then
    echo "Trying to get wandb link from $MAYBE_LOG_PATH"
    WANDB_URL=$(head -n 200 $MAYBE_LOG_PATH | grep 'https://wandb.ai.*/runs/' | cut -c 22- )
    echo $WANDB_URL
else
    WANDB_URL="unknown"
fi

NAME="polaris-upload-$JOBID"
DESCRIPTION="Uploaded output from polaris job $JOBID. \
You may be able to find the corresponding wandb run here: $WANDB_URL. \
The beaker dataset was created from $PATH_TO_UPLOAD."

echo "Uploading $PATH_TO_UPLOAD to beaker dataset named $NAME"

beaker dataset create \
    --name "$NAME" \
    --desc "$DESCRIPTION" \
    $PATH_TO_UPLOAD
