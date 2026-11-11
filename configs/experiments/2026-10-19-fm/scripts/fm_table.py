"""Per-architecture progress table for the 2026-10-19 FM training jobs.

Epochs done and per-phase timings come from the logs of every run
(preemption retry) of each finished or running experiment. Epochs done is the
highest N in "Starting training step for model trained for N complete
epochs", so an epoch counts only once its training, validation and inline
inference are all done. Finished jobs count as max_epochs; never-started jobs
count as 0. Train time is the wall time that produced kept progress, summed
over an experiment's jobs (preemption retries): each job counts from its first
training step to its exit (or now, if running), except that a preempted or
failed job counts only up to its last checkpoint save, since the work after it
is redone by the next retry. Queueing, startup and lost work are excluded, so
preemptions do not inflate it. Epoch and train-time
columns cover finished and running jobs only. The 2026-10-19 -e3sm configs are
excluded; the 2026-11-09 v2 runs (create_fm_v2_jobs.py, 50 epochs) get one
row per architecture and data arm (fm-era5, fm-balanced, fm-e3sm-balanced,
fm-spencer-balanced). Only the -maskoff v2 runs are tracked; the -maskon
runs are paused and left out. The 2026-11-11 follow-ups of the v2 runs
(create_fm_v2_followup_jobs.tracked_runs) are counted in their parent's row,
not rows of their own: the swin LR sweep (30 epochs) in the swin-v2.2-v2
fm-balanced row, the ext80 forks of both architectures and the swin e120 runs
(120 epochs; only for the LRs in E120_RUNS, so unchosen ones never count as
not submitted) in their arm's row. The ext80 forks restart the epoch count
at the fork, so their epochs are shown parent-equivalent (child epoch + 40,
finished = 120).
The v1 sfno and older swin runs are not tracked (ARCHS); their logs are not read. All
values are measured; there are no estimates.

"Not submitted" counts job names with no Beaker experiment;
"Failed / Other" counts only experiments that exist and failed or were canceled.
"Preemptions" sums, over the row's experiments, the jobs Beaker canceled
because a higher-priority job preempted them.

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
    job_status,
    list_experiments,
)
from create_fm_v2_followup_jobs import tracked_runs  # noqa: E402
from submit_fm_jobs import (  # noqa: E402
    ARCH_CHOICES,
    V2_ARCH_CHOICES,
    config_to_job_name,
    select_configs,
)

MAX_EPOCHS = 150
V2_MAX_EPOCHS = 50
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
# Checkpoint saves: at each epoch end and every checkpoint_every_n_batches. A
# preempted run resumes from the last one, so its later work is redone.
CHECKPOINT_RE = re.compile(
    r"^(\S+)Z .*Saving (?:checkpoints for epoch|latest checkpoint)", re.M
)
# Two tables with the same rows, so each fits a terminal: job counts and
# epochs, then timings.
COUNT_COLUMNS = (
    "Architecture",
    "Finished",
    "Running",
    "Queued",
    "Failed / Other",
    "Not submitted",
    "Preemptions",
    "Min epoch",
    "Max epoch",
    "Median epoch",
)
TIMING_COLUMNS = (
    "Architecture",
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


def _queued(named: NamedExperiment) -> bool:
    """A running experiment whose unfinished job has not started yet."""
    jobs = named.experiment.get("jobs") or []
    pending = [j for j in jobs if job_status(j) == "running"]
    return bool(pending) and not pending[-1].get("status", {}).get("started")


def _preemptions(named: NamedExperiment) -> int:
    """Jobs of the experiment that Beaker canceled to run a preempting job."""
    return sum(
        "preempted" in (job.get("status", {}).get("canceledFor") or "")
        for job in named.experiment.get("jobs") or []
    )


# Rows to track: per architecture, the four v2 data arms
# (row suffix "-era5" for fm-era5, none for fm-balanced, "-e3sm" for
# fm-e3sm-balanced, "-spencer" for fm-spencer-balanced). The older swin rows
# ("swin-v2.2", "swin-v2.2-res") are left out.
ARCHS = (
    "sfno-v2-era5",
    "sfno-v2",
    "sfno-v2-e3sm",
    "sfno-v2-spencer",
    "swin-v2.2-v2-era5",
    "swin-v2.2-v2",
    "swin-v2.2-v2-e3sm",
    "swin-v2.2-v2-spencer",
)
# Job-name tags of the v2 runs, longest first: they extend the old tags.
V2_TAGS = (("-nc-swin-v2.2-v2-", "swin-v2.2-v2"), ("-nc-sfno-v2-", "sfno-v2"))
V2_ARM_SUFFIXES = (
    ("-fm-era5-", "-era5"),
    ("-fm-e3sm-balanced-", "-e3sm"),
    ("-fm-spencer-balanced-", "-spencer"),
)
# One-off variant configs, kept out of run_configs/ so batch submits skip them.
VARIANTS_DIR = pathlib.Path(__file__).parent.parent / "variants"


def arch_of(name: str) -> str:
    for tag, arch in V2_TAGS:
        if tag in name:
            for arm_tag, suffix in V2_ARM_SUFFIXES:
                if arm_tag in name:
                    return f"{arch}{suffix}"
            return arch
    for arch in ("swin-v2.2-res", "swin-v2.2"):
        if f"-{arch}-" in name:
            return arch
    return "sfno"


def is_v2(arch: str) -> bool:
    return any(arch.startswith(base) for _, base in V2_TAGS)


# Follow-up runs by job name: their own max_epochs and, for forks, the
# parent epochs before their first.
FOLLOWUPS = {run.job_name: run for run in tracked_runs()}


def max_epochs_of(arch: str, name: str | None = None) -> int:
    if name in FOLLOWUPS:
        return FOLLOWUPS[name].max_epochs
    return V2_MAX_EPOCHS if is_v2(arch) else MAX_EPOCHS


def epoch_offset_of(name: str) -> int:
    return FOLLOWUPS[name].epoch_offset if name in FOLLOWUPS else 0


def _log_time(t: str) -> datetime:
    return datetime.fromisoformat(t).replace(tzinfo=UTC)


def useful_seconds(job: dict, log: str) -> float:
    """Wall seconds of one job (run) whose progress was kept: from its first
    training step to its exit (or now, if still running); a preempted or
    failed job ends at its last checkpoint save, as later work is redone.
    """
    starts = [_log_time(t) for t, _, _ in PHASE_RE.findall(log)]
    if not starts:
        return 0.0
    start = starts[0]
    status = job.get("status", {})
    if status.get("exited") and status.get("exitCode") == 0:
        end = datetime.fromisoformat(status["exited"])
    elif not status.get("exited") and not status.get("canceled"):
        end = datetime.now(UTC)
    else:
        saves = [_log_time(t) for t in CHECKPOINT_RE.findall(log)]
        saves = [t for t in saves if t >= start]
        end = saves[-1] if saves else start
    return max((end - start).total_seconds(), 0.0)


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 1440}d {minutes // 60 % 24}h {minutes % 60}m"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


# Inline inference runs the aimip evaluator(s) every epoch and all others every
# 10 epochs: for the v1 runs with `epochs: {start: 0, step: 10}`, i.e. after
# epochs 1, 11, 21, ...; for the v2 runs (the 10yr and 46yr ERA5 runs and the
# 5-day weather ensembles) with `epochs: {start: 9, step: 10}`, i.e. after
# epochs 10, 20, ..., 50.
HEAVY_INFERENCE_EVERY = 10


def heavy_inference_residue(arch: str) -> int:
    return 0 if is_v2(arch) else 1


def _is_heavy_inference(epoch: int, residue: int) -> bool:
    return epoch % HEAVY_INFERENCE_EVERY == residue


Seconds = tuple[float | None, float | None, float | None, float | None]


Intervals = tuple[list[float], list[float], list[float], list[float]]


def epoch_intervals(log: str, heavy_residue: int) -> Intervals:
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
            (heavy if _is_heavy_inference(n1, heavy_residue) else light).append(seconds)
    return full, train, light, heavy


def _run_log(job_id: str) -> str:
    grep = (
        "grep -E 'Starting (training|validation|inference) step|Resuming training"
        "|Saving (checkpoints for epoch|latest checkpoint)'"
    )
    return subprocess.run(
        f"beaker job logs {job_id} 2>/dev/null | {grep}",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout


def progress(named, arch: str, name: str | None = None) -> tuple[int, Seconds, float]:
    """(epochs done, median seconds per (epoch, training, light and heavy
    inference), useful train seconds) over the logs of every run (preemption
    retry) of the experiment, finished or not. Epochs done are
    parent-equivalent for forks (epoch_offset_of).
    """
    none: Seconds = (None, None, None, None)
    offset = epoch_offset_of(name) if name is not None else 0
    max_epochs = max_epochs_of(arch, name) + offset
    if named is None:
        return 0, none, 0.0
    jobs = named.experiment.get("jobs") or []
    started = [j for j in jobs if "started" in j.get("status", {})]
    if not started:
        return (max_epochs if named.status == OK else 0), none, 0.0
    intervals: Intervals = ([], [], [], [])
    done = 0
    useful = 0.0
    for job in started:
        log = _run_log(job["id"])
        useful += useful_seconds(job, log)
        done = max(done, *(int(m.group(1)) for m in EPOCH_RE.finditer(log)), 0)
        for acc, values in zip(
            intervals, epoch_intervals(log, heavy_inference_residue(arch))
        ):
            acc.extend(values)
    done += offset
    if named.status == OK:
        done = max_epochs
    full, train, light, heavy = intervals
    medians = (_median(full), _median(train), _median(light), _median(heavy))
    return done, medians, useful


def main() -> None:
    names = [
        config_to_job_name(c)
        for c in select_configs(None, ARCH_CHOICES)
        if "e3sm" not in c
    ]
    names += [config_to_job_name(c) for c in select_configs(None, V2_ARCH_CHOICES)]
    names += [config_to_job_name(p.name) for p in sorted(VARIANTS_DIR.glob("*.yaml"))]
    names += list(FOLLOWUPS)
    # The -maskon v2 runs are paused (stopped on Beaker) and not tracked.
    names = [
        n
        for n in dict.fromkeys(names)
        if arch_of(n) in ARCHS and not n.endswith("-maskon")
    ]
    # List per architecture by the names' longest common prefix: the broad
    # "ace2-fm-nc-" listing matches ~9k old experiments and takes ~5 min,
    # while each narrow one takes ~1 s.
    # One listing per row and job-name tag (the part before "-fm-"), so a
    # row's follow-up runs are listed apart from its v2 runs.
    by_name = {}
    for arch in ARCHS:
        arch_names = [n for n in names if arch_of(n) == arch]
        for tag in dict.fromkeys(n.partition("-fm-")[0] for n in arch_names):
            # Never empty, so never the common prefix "", which would list
            # the whole workspace.
            tag_names = [n for n in arch_names if n.startswith(f"{tag}-fm-")]
            by_name.update(newest_by_name(os.path.commonprefix(tag_names)))
    with ThreadPoolExecutor(20) as ex:
        prog = dict(
            zip(
                names,
                ex.map(
                    lambda n: progress(by_name.get(n), arch_of(n), n),
                    names,
                ),
            )
        )
    count_rows = []
    timing_rows = []
    for arch in ARCHS:
        arch_names = [n for n in names if arch_of(n) == arch]
        statuses = [
            by_name[n].status if n in by_name else "missing" for n in arch_names
        ]
        # Epoch columns cover finished and running jobs only; failed/canceled
        # or deleted jobs would pull them toward 0.
        live_pairs = [
            (n, s) for n, s in zip(arch_names, statuses) if s in (OK, "running")
        ]
        live = [n for n, _ in live_pairs]
        eps = [prog[n][0] for n in live] or [0]
        times = [prog[n][2] for n in live]

        def median_column(i: int) -> str:
            values = [s for n in live if (s := prog[n][1][i]) is not None]
            return f"{statistics.median(values):.0f}" if values else "-"

        count_rows.append(
            (
                arch,
                sum(s == OK for s in statuses),
                sum(s == "running" and not _queued(by_name[n]) for n, s in live_pairs),
                sum(s == "running" and _queued(by_name[n]) for n, s in live_pairs),
                sum(s not in (OK, "running", "missing") for s in statuses),
                sum(s == "missing" for s in statuses),
                sum(_preemptions(by_name[n]) for n in arch_names if n in by_name),
                min(eps),
                max(eps),
                f"{statistics.median(eps):g}",
            )
        )
        timing_rows.append(
            (
                arch,
                median_column(0),
                median_column(1),
                median_column(2),
                median_column(3),
                format_duration(statistics.median(times)) if times else "-",
            )
        )
    _print_table(COUNT_COLUMNS, count_rows)
    print()
    _print_table(TIMING_COLUMNS, timing_rows)


def _print_table(columns: tuple[str, ...], rows: list[tuple]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")


if __name__ == "__main__":
    main()
