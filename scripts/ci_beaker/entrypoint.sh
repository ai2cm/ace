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

cp docs/evaluator-config.yaml /workspace/evaluator-config.yaml

mkdir -p /results

COMMIT_SHA_SHORT=$(git rev-parse --short HEAD)
export WANDB_NAME="ci-evaluator-${COMMIT_SHA_SHORT}"

python -m fme.ace.evaluator /workspace/evaluator-config.yaml \
    --override \
    experiment_dir=/results \
    n_forward_steps=2800 \
    forward_steps_in_memory=30 \
    checkpoint_path=/test-data/ckpt.tar \
    loader.dataset.data_path=/test-data/data \
    loader.dataset.n_repeats=24 \
    logging.log_to_wandb=True \
    logging.project=ace-ci-tests \
    logging.entity=ai2cm \
    aggregator.log_global_mean_norm_time_series=False \
    aggregator.log_zonal_mean_images=False \
    | tee /results/out.log
