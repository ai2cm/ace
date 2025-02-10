#!/bin/bash

# Exit script if any commands fail.
set -e
set -o pipefail

# Check that the environment variables have been set correctly
for env_var in "$TMP_GITHUB_TOKEN" "$COMMIT_SHA"; do
    if [[ -z "$env_var" ]]; then
        echo >&2 "error: required environment variable $env_var is empty"
        exit 1
    fi
done

export "GITHUB_TOKEN=${TMP_GITHUB_TOKEN}"

# Configure git to use GitHub CLI as a credential helper so that we can clone private repos.
gh auth setup-git

mkdir fme && cd fme
gh repo clone ai2cm/full-model .
git checkout --quiet "$COMMIT_SHA"

pip install --no-deps -e .

yq eval '.experiment_dir="/results" | .n_forward_steps=40 | .forward_steps_in_memory=10 | .checkpoint_path="/test-default/ckpt.tar" | .loader.dataset.data_path="/test-default/data"' docs/evaluator-config.yaml > /workspace/evaluator-config.yaml

mkdir -p /results

# Change direcotry to avoid reading from fme
cd /results

exec "$@" 2>&1 | tee /results/out.log
