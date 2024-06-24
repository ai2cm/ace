#!/bin/bash

timestamp=$(date +%Y-%m-%d)

python3 generate_train_report.py \
    fv3gfs-ensemble-4deg-8layer.yaml \
    $timestamp-climsst-4deg \
    10 \
    1000000
