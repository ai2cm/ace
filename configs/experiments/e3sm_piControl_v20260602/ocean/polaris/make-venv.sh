#!/bin/bash
COMMIT=$1
ENV_PREFIX=${2:-"/eagle/E3SMinput/elynnwu/fme-env"}
ENVIRONMENT_PATH=$ENV_PREFIX/$COMMIT
SCRATCH=/eagle/E3SMinput/elynnwu/scratch
if [ -e "$ENVIRONMENT_PATH/bin/python" ]; then
    echo "$ENVIRONMENT_PATH exists, reusing the env."
else
    rm -rf $SCRATCH/ace-pbs-env/temp/
    mkdir -p $SCRATCH/ace-pbs-env/temp/
    cd $SCRATCH/ace-pbs-env/temp/
    git clone git@github.com:ai2cm/ace.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository git@github.com:ai2cm/ace.git"
        exit 1
    fi

    cd $SCRATCH/ace-pbs-env/temp/ace
    git checkout "$COMMIT"
    if [ $? -ne 0 ]; then
        echo "Failed to checkout commit: $COMMIT"
        exit 1
    fi

    echo "Creating environment at $ENVIRONMENT_PATH"
    module use /soft/modulefiles
    module load conda
    conda create -p $ENVIRONMENT_PATH python=3.11 pip -y
    conda run --no-capture-output -p $ENVIRONMENT_PATH python -m pip install uv
    conda run --no-capture-output -p $ENVIRONMENT_PATH uv pip install -c constraints.txt .[dev,docs]
    rm -rf $SCRATCH/ace-pbs-env/temp/ace
fi

echo $ENVIRONMENT_PATH
