"""Per-architecture progress table for the 2026-10-19 FM training jobs.

Epochs done and per-phase timings come from the logs of every run
(preemption retry) of each finished or running experiment. Epochs done is the
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
from _beaker_listing import (  # noqa: E402
    OK,
    NamedExperiment,
    canonical_name,
    experiment_status,
    list_experiments,
)
from submit_fm_jobs import (  # noqa: E402
    ARCH_CHOICES,
    config_to_job_name,
    select_configs,
)

MAX_EPOCHS = 150
# "Starting training step for model trained for N complete epochs" marks that
# epochs 0..N-1 are done, including their validation and inline inference.
EPOCH_RE = re.compile(r"Starting training step for model trained for (\d+) complete")
# Phase starts per epoch. A full epoch runs from one training start to the
# next; training from a training start to the next validation start; inline
# inference from an inference start to the next training start (it includes
# the checkpoint save). A resumed run's first training start is mid-epoch, so
# its epoch and training intervals are skipped.
PHASE_RE = re.compile(
    r"^(\S+)Z .*Starting (training|validation|inference) step for model trained "
    r"for (\d+)",
    re.M,
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
    "s/epoch (median) (train)",
    "s/epoch (median) (inference, every epoch)",
    "s/epoch (median) (inference, every 10)",
    "Train time (median)",
)


def _submitted(experiment: dict) -> str:
    """When the experiment was submitted: its first job's creation time.

    Not the latest job's, which moves forward on every preemption retry.
    """
    jobs = experiment.get("jobs") or []
    created = [job.get("status", {}).get("created", "") for job in jobs]
    return min(created, default="")


def newest_by_name(text: str) -> dict[str, NamedExperiment]:
    """The newest experiment for each canonical job name matching ``text``.

    Unlike ``_beaker_listing.experiments_by_name`` (which prefers the best
    status, as the submit scripts need), a resubmitted job always replaces the
    older experiment of the same name, so the table tracks the latest run.
    """
    newest: dict[str, dict] = {}
    for experiment in list_experiments(text=text):
        name = canonical_name(experiment["name"])
        if name not in newest or _submitted(experiment) > _submitted(newest[name]):
            newest[name] = experiment
    return {
        name: NamedExperiment(name, experiment_status(e), e)
        for name, e in newest.items()
    }


ARCHS = ("sfno", "swin-v2.2", "swin-v2.2-res")
# One-off variant configs, kept out of run_configs/ so batch submits skip them.
VARIANTS_DIR = pathlib.Path(__file__).parent / "variants"


def arch_of(name: str) -> str:
    for arch in ("swin-v2.2-res", "swin-v2.2"):
        if f"-{arch}-" in name:
            return arch
    return "sfno"


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


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


# Inline inference runs the two aimip evaluators every epoch and all others
# with `epochs: {start: 0, step: 10}`, i.e. after epochs 1, 11, 21, ...
HEAVY_INFERENCE_EVERY = 10


def _is_heavy_inference(epoch: int) -> bool:
    return epoch % HEAVY_INFERENCE_EVERY == 1


Seconds = tuple[float | None, float | None, float | None, float | None]


Intervals = tuple[list[float], list[float], list[float], list[float]]


def epoch_intervals(log: str) -> Intervals:
    """Wall seconds of each (full epoch, training, inline inference on light
    epochs, inline inference on heavy every-10th epochs) in one run's log.
    """
    events = [
        (datetime.fromisoformat(t), phase, int(n))
        for t, phase, n in PHASE_RE.findall(log)
    ]
    starts = [(t, n) for t, phase, n in events if phase == "training"]
    if RESUME_MARK in log and starts:
        resumed_at = starts[0][0]
        starts = starts[1:]
    else:
        resumed_at = None
    full = [
        (t2 - t1).total_seconds()
        for (t1, n1), (t2, n2) in zip(starts, starts[1:])
        if n2 == n1 + 1
    ]
    train: list[float] = []
    light: list[float] = []
    heavy: list[float] = []
    for (t1, p1, n1), (t2, p2, n2) in zip(events, events[1:]):
        if p1 == "training" and p2 == "validation" and n2 == n1 + 1:
            if t1 != resumed_at:
                train.append((t2 - t1).total_seconds())
        elif p1 == "inference" and p2 == "training" and n2 == n1:
            seconds = (t2 - t1).total_seconds()
            (heavy if _is_heavy_inference(n1) else light).append(seconds)
    return full, train, light, heavy


def _run_log(job_id: str) -> str:
    grep = "grep -E 'Starting (training|validation|inference) step|Resuming training'"
    return subprocess.run(
        f"beaker job logs {job_id} 2>/dev/null | {grep}",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout


def progress(named) -> tuple[int, Seconds]:
    """(epochs done, median seconds per (epoch, training, light and heavy
    inference)) over the logs of every run (preemption retry) of the
    experiment, finished or not.
    """
    none: Seconds = (None, None, None, None)
    if named is None:
        return 0, none
    jobs = named.experiment.get("jobs") or []
    started = [j for j in jobs if "started" in j.get("status", {})]
    if not started:
        return (MAX_EPOCHS if named.status == OK else 0), none
    intervals: Intervals = ([], [], [], [])
    done = 0
    for job in started:
        log = _run_log(job["id"])
        done = max(done, *(int(m.group(1)) for m in EPOCH_RE.finditer(log)), 0)
        for acc, values in zip(intervals, epoch_intervals(log)):
            acc.extend(values)
    if named.status == OK:
        done = MAX_EPOCHS
    full, train, light, heavy = intervals
    return done, (_median(full), _median(train), _median(light), _median(heavy))


def main() -> None:
    names = [
        config_to_job_name(c)
        for c in select_configs(None, ARCH_CHOICES)
        if "e3sm" not in c
    ] + [config_to_job_name(p.name) for p in sorted(VARIANTS_DIR.glob("*.yaml"))]
    # List per architecture by the names' longest common prefix: the broad
    # "ace2-fm-nc-" listing matches ~9k old experiments and takes ~5 min,
    # while each narrow one takes ~1 s.
    by_name = {}
    for arch in ARCHS:
        arch_names = [n for n in names if arch_of(n) == arch]
        by_name.update(newest_by_name(os.path.commonprefix(arch_names)))
    with ThreadPoolExecutor(20) as ex:
        prog = dict(zip(names, ex.map(lambda n: progress(by_name.get(n)), names)))
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for arch in ARCHS:
        arch_names = [n for n in names if arch_of(n) == arch]
        statuses = [
            by_name[n].status if n in by_name else "missing" for n in arch_names
        ]
        # Epoch columns cover finished and running jobs only; failed/canceled
        # or deleted jobs would pull them toward 0.
        live = [n for n, st in zip(arch_names, statuses) if st in (OK, "running")]
        eps = [prog[n][0] for n in live] or [0]
        times = [train_seconds(by_name[n]) for n in live]

        def median_column(i: int) -> str:
            values = [s for n in live if (s := prog[n][1][i]) is not None]
            return f"{statistics.median(values):.0f}" if values else "-"

        row = (
            arch,
            sum(s == OK for s in statuses),
            sum(s == "running" for s in statuses),
            sum(s not in (OK, "running") for s in statuses),
            min(eps),
            max(eps),
            f"{statistics.median(eps):g}",
            median_column(0),
            median_column(1),
            median_column(2),
            median_column(3),
            format_duration(statistics.median(times)) if times else "-",
        )
        print("| " + " | ".join(str(v) for v in row) + " |")


if __name__ == "__main__":
    main()
