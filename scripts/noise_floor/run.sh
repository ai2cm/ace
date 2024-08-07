#!/bin/bash

timestamp=$(date +%Y-%m-%d)

# You can comment out this first step if the data is already generated
python3 generate_stats.py fv3gfs-ensemble-4deg-8layer.yaml
python3 generate_train_report.py \
    fv3gfs-ensemble-4deg-8layer.yaml \
    $timestamp-climsst-4deg \
    10 \
    1000000
