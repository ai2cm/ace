#!/bin/bash

CONFIG=$1

set -e

argo submit generate-stats-argo-workflow.yaml \
     -p python_script="$(< generate_stats.py)" \
     -p config="$(< ${CONFIG})"
