#!/bin/bash
# Shim script that calls the centralized uncoupled_train.sh wrapper
# This maintains backward compatibility with existing workflows

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(git rev-parse --show-prefix)

# Remove trailing slash if present
SCRIPT_DIR=${SCRIPT_DIR%/}

# Remove the script filename from the path to get the experiment directory
EXPERIMENT_DIR=${SCRIPT_DIR}

exec "$REPO_ROOT/configs/coupled_job_runner/uncoupled_train.sh" "$EXPERIMENT_DIR" "$@"
