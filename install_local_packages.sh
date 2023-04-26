#!/bin/bash

# install all packages that are defined within this repo using --no-deps flag

LOCAL_PACKAGES="models/FourCastNet \
    models/fcn-mip \
    models/fcn-mip/external/torch_sht \
    models/fcn-mip/external/modulus-core \
    fme"

for PACKAGE in $LOCAL_PACKAGES; do
    python -m pip install --no-deps -e $PACKAGE
done
