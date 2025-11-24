#!/bin/bash

set -e

if [ ! -d ./ocean_emulators ]; then
    git clone git@github.com:m2lines/ocean_emulators.git # forked from m2lines/ocean_emulators
fi

cd ocean_emulators
echo "ocean_emulators SHA: $(git rev-parse HEAD)"
uv pip install -e ".[dev]"
