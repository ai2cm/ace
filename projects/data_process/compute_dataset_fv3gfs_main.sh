#!/bin/bash

argo submit compute_dataset_fv3gfs_argo_workflow.yaml -p python_script="$(< compute_dataset_fv3gfs.py)"
