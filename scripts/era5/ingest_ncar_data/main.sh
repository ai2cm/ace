#!/bin/bash

argo submit argo_workflow.yaml \
    -p python_script="$(< ingest_single_variable.py)" \
    -p variables="$(< variables.json)" \
    -p script_flags="--gcs-dir gs://vcm-ml-raw-flexible-retention/2024-03-11-era5-025deg-2D-variables-from-NCAR"
