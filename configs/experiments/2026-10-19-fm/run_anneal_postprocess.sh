#!/bin/bash
# Post-process the anneal sweep once its 30 training runs have succeeded:
#
#   1. refresh wandb_to_beaker_map.json (update_beaker_map.py);
#   2. make and upload checkpoint C of every run (make_soup_checkpoints.py:
#      soup of ema_ckpt_0016..0020 -> Beaker dataset {run}-ckptC, recorded in
#      anneal_soups.json);
#   3. submit J1/J3/J4 for checkpoints C and B, plus any missing P job
#      (submit_anneal_eval_jobs.py --skip-if-in-beaker).
#
# Stops before step 2 unless all 30 runs have succeeded; ALLOW_PARTIAL=1
# goes on with the finished ones (the others are skipped with a message, and
# a rerun later fills them in: every step is idempotent).
# Extra arguments go to both python scripts' selection flags, e.g.
# `--arms a1 --seeds 0`. Set DRY_RUN=1 to only print what would be done.
#
# Usage:
#   ./run_anneal_postprocess.sh [--arms ...] [--seeds ...]
#   then commit and push wandb_to_beaker_map.json and anneal_soups.json.

set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.virtualenvs/ace/bin:$PATH"
export PYTHONPATH="$(git rev-parse --show-toplevel)"
DRY=()
[[ "${DRY_RUN:-0}" == "1" ]] && DRY=(--dry-run)

python update_beaker_map.py "${DRY[@]}"

python - "$@" <<'PY'
import os, sys
from _inference_common import anneal_run_names, finished_runs
ready, skipped = finished_runs(anneal_run_names())
print(f"{len(ready)}/{len(ready) + len(skipped)} anneal runs succeeded")
for run, reason in sorted(skipped.items()):
    print(f"  not finished: {run} ({reason})")
if skipped and os.environ.get("ALLOW_PARTIAL") != "1":
    sys.exit("Not all anneal runs have succeeded; set ALLOW_PARTIAL=1 to go on.")
PY

python make_soup_checkpoints.py "${DRY[@]}" "$@"
python submit_anneal_eval_jobs.py --ckpts C,B,P --families J1,J3,J4 \
  --skip-if-in-beaker "${DRY[@]}" "$@"

echo "Done. Commit and push wandb_to_beaker_map.json and anneal_soups.json."
