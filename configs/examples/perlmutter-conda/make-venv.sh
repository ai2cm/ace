#!/bin/bash
COMMIT=$1
ENV_PREFIX=${2:-"$SCRATCH/fme-conda-envs"}
ENVIRONMENT_PATH=$ENV_PREFIX/$COMMIT

if [ -e "$ENVIRONMENT_PATH/bin/python" ]; then
    echo "$ENVIRONMENT_PATH exists, reusing the env."
else
    rm -rf $SCRATCH/ace-slurm-env/temp/
    mkdir -p $SCRATCH/ace-slurm-env/temp/
    cd $SCRATCH/ace-slurm-env/temp/
    git clone git@github.com:ai2cm/full-model.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone repository git@github.com:ai2cm/full-model.git"
        exit 1
    fi

    cd $SCRATCH/ace-slurm-env/temp/full-model
    git checkout "$COMMIT"
    if [ $? -ne 0 ]; then
        echo "Failed to checkout commit: $COMMIT"
        exit 1
    fi

    echo "Creating environment at $ENVIRONMENT_PATH"
    module load python
    conda create -p $ENVIRONMENT_PATH python=3.11 pip -y
    conda run --no-capture-output -p $ENVIRONMENT_PATH python -m pip install uv
    conda run --no-capture-output -p $ENVIRONMENT_PATH uv pip install -c constraints.txt .[dev,docs]
    rm -rf $SCRATCH/ace-slurm-env/temp/full-model
fi

echo $ENVIRONMENT_PATH
