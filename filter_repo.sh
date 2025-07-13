#!/usr/bin/env bash

set -e

git clone . filtered_repo/

cd filtered_repo/

git checkout main

git remote remove origin

git filter-repo --force --refs "refs/heads/main" \
    --path .github/workflows/tests.yaml \
    --path .github/workflows/pre-commit.yml \
    --path .github/workflows/docs.yaml \
    --path .github/pull_request_template.md \
    --path docker \
    --path fme/ace/aggregator \
    --path fme/ace/data_loading \
    --path fme/ace/inference \
    --path fme/ace/models/healpix \
    --path fme/ace/models/makani \
    --path fme/ace/models/modulus \
    --path fme/ace/models/ocean \
    --path fme/ace/models/__init__.py \
    --path fme/ace/registry/__init__.py \
    --path fme/ace/registry/hpx.py \
    --path fme/ace/registry/m2lines.py \
    --path fme/ace/registry/prebuilt.py \
    --path fme/ace/registry/registry.py \
    --path fme/ace/registry/sfno.py \
    --path fme/ace/registry/stochastic_sfno.py \
    --path fme/ace/registry/test_hpx.py \
    --path fme/ace/registry/test_m2lines.py \
    --path fme/ace/registry/test_sfno.py \
    --path fme/ace/registry/test_stochastic_sfno.py \
    --path fme/ace/stepper \
    --path fme/ace/testing \
    --path fme/ace/train \
    --path fme/ace/__init__.py \
    --path fme/ace/evaluator.py \
    --path fme/ace/LICENSE \
    --path fme/ace/requirements.py \
    --path fme/ace/run-train-and-inference.sh \
    --path fme/ace/test_ocean_train.py \
    --path fme/ace/test_train.py \
    --path fme/ace/validate_config.py \
    --path fme/core \
    --path fme/core/__init__.py \
    --path fme/__init__.py \
    --path fme/require_gpu.py \
    --path fme/sht_fix.py \
    --path fme/test_harmonics.py \
    --path fme/test_torch.py \
    --path docs \
    --path scripts/data_process \
    --path scripts/era5 \
    --path scripts/manual_backwards_compatibility \
    --path scripts/monthly_data \
    --path scripts/noise_floor \
    --path scripts/README.md \
    --path .gitignore \
    --path .pre-commit-config.yaml \
    --path ACE-logo.png \
    --path analysis-deps.txt \
    --path conftest.py \
    --path constraints.txt \
    --path LICENSE \
    --path Makefile \
    --path PACKAGE_README.md \
    --path pyproject.toml \
    --path README.md \
    --path requirements-deploy.txt \
    --path requirements-dev.txt \
    --path requirements-healpix.txt \
    --path requirements.txt

git reflog expire --expire=now --expire-unreachable=now --all && git gc --prune=now --aggressive

cd ..
diff \
  <(git ls-tree -r main --name-only) \
  <(cd filtered_repo && git ls-tree -r main --name-only) \
  | grep '^< ' | sed 's/< //g' > files_excluded_by_filter.txt
