"""Per-architecture progress table for the 2026-10-19 FM training jobs.

Epochs done per job come from the latest *started* beaker job's log: the max of
"trained for N complete epochs" (checkpoint saves, incl. after a resume) and
"model trained for N epochs" (validation/inference starts). Finished jobs
count as max_epochs; never-started jobs count as 0. The -e3sm configs are
excluded.

Usage:
    python fm_table.py
"""

import pathlib
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _beaker_listing import OK, fetch_experiments_by_name  # noqa: E402
from submit_fm_jobs import (  # noqa: E402
    ARCH_CHOICES,
    config_to_job_name,
    select_configs,
)

MAX_EPOCHS = 150
EPOCH_RE = re.compile(r"trained for (\d+) (?:complete )?epochs")
# Wall time between consecutive validation starts = one full epoch, including
# validation and inline inference.
VAL_RE = re.compile(
    r"^(\S+)Z .*Starting validation step for model trained for (\d+) epochs", re.M
)
# Estimated wall seconds per epoch (train + validation + inline inference).
# - sfno: measured ~2450 s.
# - swin-v2.2 dm1e256 compiled: roughly SFNO speed per the 2026-10-26 speed
#   study benchmarks (64 vs 57 samples/s/GPU with batched EMA), so ~2800 s.
EST_S_PER_EPOCH = {"sfno": 2450, "swin-v2.2": 2800}
COLUMNS = (
    "Architecture",
    "Finished",
    "Running or queued",
    "Failed/other",
    "Min epoch",
    "Max epoch",
    "Median epoch",
    "s/epoch (median)",
    "est. s/epoch",
)


def arch_of(name: str) -> str:
    return "swin-v2.2" if "swin-v2.2" in name else "sfno"


def progress(named) -> tuple[int, float | None]:
    """(epochs done, median wall seconds per epoch in the last started job)."""
    if named is None:
        return 0, None
    # A preempted experiment's newest job may still be queued, so read the
    # last job that actually ran.
    jobs = named.experiment.get("jobs") or []
    started = [j for j in jobs if "started" in j.get("status", {})]
    if not started:
        return (MAX_EPOCHS if named.status == OK else 0), None
    grep = "grep -E 'trained for [0-9]+ (complete )?epochs'"
    proc = subprocess.run(
        f"beaker job logs {started[-1]['id']} 2>/dev/null | {grep}",
        shell=True,
        capture_output=True,
        text=True,
    )
    done = max((int(m.group(1)) for m in EPOCH_RE.finditer(proc.stdout)), default=0)
    if named.status == OK:
        done = MAX_EPOCHS
    marks = [
        (datetime.fromisoformat(t), int(n)) for t, n in VAL_RE.findall(proc.stdout)
    ]
    diffs = [
        (t2 - t1).total_seconds()
        for (t1, n1), (t2, n2) in zip(marks, marks[1:])
        if n2 == n1 + 1
    ]
    return done, (statistics.median(diffs) if diffs else None)


def main() -> None:
    names = [
        config_to_job_name(c)
        for c in select_configs(None, ARCH_CHOICES)
        if "e3sm" not in c
    ]
    by_name = fetch_experiments_by_name(text="ace2-fm-nc-")
    with ThreadPoolExecutor(10) as ex:
        prog = dict(zip(names, ex.map(lambda n: progress(by_name.get(n)), names)))
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for arch in ("sfno", "swin-v2.2"):
        arch_names = [n for n in names if arch_of(n) == arch]
        statuses = [
            by_name[n].status if n in by_name else "missing" for n in arch_names
        ]
        eps = [prog[n][0] for n in arch_names]
        spe = [s for n in arch_names if (s := prog[n][1]) is not None]
        row = (
            arch,
            sum(s == OK for s in statuses),
            sum(s == "running" for s in statuses),
            sum(s not in (OK, "running") for s in statuses),
            min(eps),
            max(eps),
            f"{statistics.median(eps):g}",
            f"{statistics.median(spe):.0f}" if spe else "-",
            EST_S_PER_EPOCH[arch],
        )
        print("| " + " | ".join(str(v) for v in row) + " |")


if __name__ == "__main__":
    main()
