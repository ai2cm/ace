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
    --path fme/ace \
    --path fme/core \
    --path fme/require_gpu.py \
    --path fme/sht_fix.py \
    --path fme/test_harmonics.py \
    --path deploy-requirements.txt \
    --path dev-requirements.txt \
    --path requirements.txt \
    --path LICENSE \
    --path README.md \
    --path PACKAGE_README.md \
    --path pyproject.toml \
    --path docs \
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
