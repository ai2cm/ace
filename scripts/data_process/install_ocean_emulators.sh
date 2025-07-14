#!/bin/bash

set -e

if [ ! -d ./ocean_emulators ]; then
    git clone git@github.com:jpdunc23/ocean_emulators.git # forked from m2lines/ocean_emulators
fi

cd ocean_emulators && git checkout cm4-preprocessing
echo "ocean_emulators SHA: $(git rev-parse HEAD)"
uv pip install -e ".[dev]"
