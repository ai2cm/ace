#!/usr/bin/env bash

set -e

git clone . filtered_repo/

cd filtered_repo/

git checkout main

git remote remove origin

git filter-repo --force --refs "refs/heads/main" \
    --path .github/workflows/tests.yaml \
    --path .github/workflows/pre-commit.yml \
    --path .github/pull_request_template.md \
    --path docker \
    --path fme/fme/ace \
    --path fme/fme/core \
    --path fme/fme/require_gpu.py \
    --path fme/fme/sht_fix.py \
    --path fme/fme/test_harmonics.py \
    --path fme/deploy-requirements.txt \
    --path fme/dev-requirements.txt \
    --path fme/requirements.txt \
    --path fme/LICENSE \
    --path fme/README.md \
    --path fme/pyproject.toml \
    --path fme/docs \
    --path scripts/data_process \
    --path scripts/era5 \
    --path scripts/monthly_data \
    --path scripts/manual_backwards_compatibility \
    --path scripts/README.md \
    --path .gitignore \
    --path .pre-commit-config.yaml \
    --path analysis-deps.txt \
    --path conftest.py \
    --path constraints.txt \
    --path LICENSE \
    --path Makefile \
    --path README.md

git reflog expire --expire=now --expire-unreachable=now --all && git gc --prune=now --aggressive
