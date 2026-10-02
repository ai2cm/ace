#!/bin/bash
# Runs inside the Beaker job. Interleaves the fme.core.benchmark.run CUDA-timer
# benchmarks on base (main + A/B config, no PR) and PR (base + merge of PR #1519)
# on the same GPU. The base tree is the base commit with this branch's benchmark
# files laid on top, so base vs PR differs by exactly the PR's fme/ changes.
# This script is copied to /tmp and run from there because it checks out other
# commits mid-run.

set -euo pipefail

BASE_SHA=8932a8612c75a82eee2dab3e06eb7b07be554bb4  # full SHA: gantry's clone is shallow and fetch-by-SHA needs it
ROUNDS=3
ITERS=100
BENCHMARKS="csfno_block_disco csfno_block"
BENCH_FILES="fme/core/models/conditional_sfno/benchmark.py fme/core/benchmark/testdata/csfno_block_disco-regression.pt"
EXPECTED_FME_DIFF="fme/ace/models/healpix/healpix_activations.py
fme/ace/models/healpix/test_healpix_activations.py
fme/ace/models/ocean/m2lines/activations.py
fme/ace/models/ocean/m2lines/samudra.py
fme/ace/models/ocean/m2lines/test_activations.py
fme/ace/models/ocean/m2lines/test_samudra.py
fme/core/benchmark/test_timer.py
fme/core/benchmark/timer.py
fme/core/disco/_fft.py
fme/core/disco/test_fft.py
fme/core/models/conditional_sfno/sfnonet.py
fme/core/models/conditional_sfno/test_sfnonet.py
fme/fft.py
fme/test_fft.py"

cd "$(git rev-parse --show-toplevel)"
PR_SHA=$(git rev-parse HEAD)
git cat-file -e "$BASE_SHA^{commit}" 2>/dev/null || git fetch -q --depth=1 origin "$BASE_SHA" || git fetch -q --unshallow origin

checkout_variant() {
  git checkout -q -f "$PR_SHA"
  if [[ "$1" == "base" ]]; then
    git checkout -q "$BASE_SHA"
    git checkout -q "$PR_SHA" -- $BENCH_FILES
    actual=$(git diff --name-only "$PR_SHA" -- fme)
    if [[ "$actual" != "$EXPECTED_FME_DIFF" ]]; then
      echo "base tree differs from PR tree by unexpected files:"; echo "$actual"; exit 1
    fi
  fi
  python -c "import fme, os, sys; p = os.path.abspath(fme.__file__); print('fme from', p); sys.exit(0 if p.startswith(os.getcwd()) else 1)"
}

nvidia-smi --query-gpu=name,driver_version --format=csv
for round in $(seq 1 $ROUNDS); do
  for variant in base pr; do
    checkout_variant "$variant"
    for name in $BENCHMARKS; do
      echo "=== round=$round variant=$variant benchmark=$name tree=$(git rev-parse --short HEAD)"
      python -m fme.core.benchmark.run --name "$name" --iters $ITERS --child filter \
        --output-dir "/results/$variant-r$round"
    done
  done
done
git checkout -q -f "$PR_SHA"
