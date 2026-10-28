"""Per-architecture progress table for the 2026-10-19 FM training jobs.

Epochs done per job come from the latest *started* beaker job's log: the
highest N in "Starting training step for model trained for N complete
epochs", so an epoch counts only once its training, validation and inline
inference are all done. Finished jobs count as max_epochs; never-started jobs
count as 0. Train time is the running wall time summed over an experiment's
jobs (preemption retries included, queueing excluded). Epoch and train-time
columns cover finished and running jobs only. The -e3sm configs are excluded.
All values are measured; there are no estimates.

Usage:
    python fm_table.py
"""

import os
import pathlib
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _beaker_listing import OK, fetch_experiments_by_name  # noqa: E402
from submit_fm_jobs import (  # noqa: E402
    ARCH_CHOICES,
    config_to_job_name,
    select_configs,
)

MAX_EPOCHS = 150
# "Starting training step for model trained for N complete epochs" marks that
# epochs 0..N-1 are done, including their validation and inline inference.
EPOCH_RE = re.compile(r"Starting training step for model trained for (\d+) complete")
# Wall time between consecutive training starts = one full epoch, including
# validation and inline inference. A resumed run's first training start is
# mid-epoch, so that interval is skipped.
TRAIN_START_RE = re.compile(
    r"^(\S+)Z .*Starting training step for model trained for (\d+) complete", re.M
)
RESUME_MARK = "Resuming training from"
COLUMNS = (
    "Architecture",
    "Finished",
    "Running or queued",
    "Failed/other",
    "Min epoch",
    "Max epoch",
    "Median epoch",
    "s/epoch (median)",
    "Train time (median)",
)


def arch_of(name: str) -> str:
    return "swin-v2.2" if "swin-v2.2" in name else "sfno"


def train_seconds(named) -> float:
    """Wall seconds the experiment's jobs have spent running, summed over all
    attempts (preemption retries); time spent queued is excluded.
    """
    now = datetime.now(UTC)
    total = 0.0
    for job in named.experiment.get("jobs") or []:
        status = job.get("status", {})
        if "started" not in status:
            continue
        end = status.get("exited") or status.get("canceled")
        t_end = datetime.fromisoformat(end) if end else now
        total += (t_end - datetime.fromisoformat(status["started"])).total_seconds()
    return total


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 1440}d {minutes // 60 % 24}h {minutes % 60}m"


def progress(named) -> tuple[int, float | None]:
    """(epochs done, median wall seconds per epoch in the last started job)."""
    if named is None:
        return 0, None
    # A preempted experiment's newest job may still be queued, so read the
    # last job that actually ran.
    jobs = named.experiment.get("jobs") or []
    started = [j for j in jobs if "started" in j.get("status", {})]
    if named.status == OK or not started:
        return (MAX_EPOCHS if named.status == OK else 0), None
    grep = "grep -E 'Starting training step|Resuming training from'"
    proc = subprocess.run(
        f"beaker job logs {started[-1]['id']} 2>/dev/null | {grep}",
        shell=True,
        capture_output=True,
        text=True,
    )
    done = max((int(m.group(1)) for m in EPOCH_RE.finditer(proc.stdout)), default=0)
    marks = [
        (datetime.fromisoformat(t), int(n))
        for t, n in TRAIN_START_RE.findall(proc.stdout)
    ]
    if RESUME_MARK in proc.stdout:
        marks = marks[1:]
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
    # List per architecture by the names' longest common prefix: the broad
    # "ace2-fm-nc-" listing matches ~9k old experiments and takes ~5 min,
    # while each narrow one takes ~1 s.
    by_name = {}
    for arch in ("sfno", "swin-v2.2"):
        arch_names = [n for n in names if arch_of(n) == arch]
        by_name.update(fetch_experiments_by_name(text=os.path.commonprefix(arch_names)))
    with ThreadPoolExecutor(10) as ex:
        prog = dict(zip(names, ex.map(lambda n: progress(by_name.get(n)), names)))
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for arch in ("sfno", "swin-v2.2"):
        arch_names = [n for n in names if arch_of(n) == arch]
        statuses = [
            by_name[n].status if n in by_name else "missing" for n in arch_names
        ]
        # Epoch columns cover finished and running jobs only; failed/canceled
        # or deleted jobs would pull them toward 0.
        live = [n for n, st in zip(arch_names, statuses) if st in (OK, "running")]
        eps = [prog[n][0] for n in live] or [0]
        times = [train_seconds(by_name[n]) for n in live]
        # Speed of unfinished, healthy jobs only: finished jobs skip the log
        # fetch, and failed/canceled ones are not training.
        spe = [
            s
            for n, st in zip(arch_names, statuses)
            if st == "running" and (s := prog[n][1]) is not None
        ]
        row = (
            arch,
            sum(s == OK for s in statuses),
            sum(s == "running" for s in statuses),
            sum(s not in (OK, "running") for s in statuses),
            min(eps),
            max(eps),
            f"{statistics.median(eps):g}",
            f"{statistics.median(spe):.0f}" if spe else "-",
            format_duration(statistics.median(times)) if times else "-",
        )
        print("| " + " | ".join(str(v) for v in row) + " |")


if __name__ == "__main__":
    main()
