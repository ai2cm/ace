#!/bin/bash
# Shim script that calls the centralized evaluate.sh wrapper
# This maintains backward compatibility with existing workflows

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <config_subdirectory>"
  echo "  - <config_subdirectory>: Subdirectory containing the evaluator config files (e.g., coupled/v2025-06-03-fto)"
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(git rev-parse --show-prefix)

# Remove trailing slash if present
SCRIPT_DIR=${SCRIPT_DIR%/}

# Parse the config_subdirectory argument
CONFIG_SUBDIR_ARG="$1"

# Split into mode (coupled/uncoupled) and subdir
# Example: "coupled/v2025-06-03-fto" -> MODE="coupled", CONFIG_SUBDIR="v2025-06-03-fto"
MODE=$(echo "$CONFIG_SUBDIR_ARG" | cut -d'/' -f1)
CONFIG_SUBDIR=$(echo "$CONFIG_SUBDIR_ARG" | cut -d'/' -f2-)

# Construct the experiment directory path
EXPERIMENT_DIR="${SCRIPT_DIR}/${MODE}"

exec "$REPO_ROOT/configs/coupled_job_runner/evaluate.sh" "$EXPERIMENT_DIR" "$CONFIG_SUBDIR"
