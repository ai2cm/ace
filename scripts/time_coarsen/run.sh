
#!/bin/bash

set -e

SCRIPT_PATH=$(git rev-parse --show-prefix)  # relative to the root of the repository
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT"

run_coarsen() {
  local config_filename="$1"
  local job_name="$2"
  local CONFIG_PATH="$SCRIPT_PATH/$config_filename"

  # Extract additional args from config header
  local extra_args=()
  while IFS= read -r line; do
    [[ "$line" =~ ^#\ arg:\ (.*) ]] && extra_args+=(${BASH_REMATCH[1]})
  done < "$CONFIG_PATH"

  gantry run \
    --name "$job_name" \
    --description 'Run ACE dataset time coarsening' \
    --beaker-image "$(cat $REPO_ROOT/latest_deps_only_image.txt)" \
    --workspace ai2/climate-titan \
    --priority urgent \
    --not-preemptible \
    --cluster ai2/jupiter \
    --cluster ai2/ceres \
    --env GOOGLE_APPLICATION_CREDENTIALS=/tmp/google_application_credentials.json \
    --dataset-secret google-credentials:/tmp/google_application_credentials.json \
    --gpus 0 \
    --shared-memory 400GiB \
    --weka climate-default:/climate-default \
    --budget ai2/climate \
    --system-python \
    --install "pip install --no-deps ." \
    "${extra_args[@]}" \
    -- python $SCRIPT_PATH/time_coarsen.py "$CONFIG_PATH"
}

base_name="time-coarsen"

run_coarsen "c96-shield.yaml" "$base_name-c96-shield"
