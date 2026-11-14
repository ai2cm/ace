"""Progress table for the long fine-tunes of the 2026-11-12 FM ft sweep.

Tracks only the 36 long runs of create_fm_v2_finetune_jobs.ft_long_runs (the
ext80 fm-spencer-balanced runs fine-tuned for 32 and for 128 epochs; ERA5
mixes 25, 70, 100 x 3 seeds at LR 1e-4), one row per architecture and length,
9 runs each, the SFNO rows first: sfno-v2-ft-e32, sfno-v2-ft-e128,
swin-v2.2-v2-ft-e32, swin-v2.2-v2-ft-e128. The finished 8-epoch ft runs and
the runs they descend from (the 2026-10-19 runs, the 2026-11-09 v2 runs and
their 2026-11-11 follow-ups) are no longer tracked; their logs are not read.

Epochs done and per-phase timings come from the logs of every run
(preemption retry) of each finished or running experiment. Epochs done is the
highest N in "Starting training step for model trained for N complete
epochs", so an epoch counts only once its training, validation and inline
inference are all done. Finished jobs count as their run's max_epochs (32 or
128, from the run object); never-started jobs count as 0.
Train time is the wall time that produced kept progress,
summed over an experiment's jobs (preemption retries): each job counts from
its first training step to its exit (or now, if running), except that a
preempted or failed job counts only up to its last checkpoint save, since the
work after it is redone by the next retry. Queueing, startup and lost work are
excluded, so preemptions do not inflate it. Epoch columns cover unfinished
(running or queued) jobs only, "-" when none; train-time and timing columns
cover finished and running jobs. The "inference, every 10" column is the
inline inference after epochs 10, 20, ..., when the every-10th-epoch entries
run too; "-" until a run of the row has finished one. All values are measured;
there are no estimates.

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
from create_fm_v2_finetune_jobs import ft_long_runs  # noqa: E402
from create_fm_v2_followup_jobs import SFNO  # noqa: E402

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


# Tracked runs by job name, and their rows: one per architecture and length,
# in ft_long_runs order (the SFNO rows first, each arch's shorter runs first).
FT_RUNS = {run.job_name: run for run in ft_long_runs()}


def arch_of(name: str) -> str:
    """Row of the run named `name`, e.g. sfno-v2-ft-e32 (its inference arch,
    _inference_common._add_ft_archs).
    """
    run = FT_RUNS[name]
    base = "sfno-v2" if run.arch == SFNO else "swin-v2.2-v2"
    return f"{base}-ft-e{run.max_epochs}"


ARCHS = tuple(dict.fromkeys(arch_of(name) for name in FT_RUNS))


def max_epochs_of(name: str) -> int:
    return FT_RUNS[name].max_epochs


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


# Inline inference runs the aimip evaluator every epoch and all others (the
# 10yr and 46yr ERA5 runs and the 5-day weather ensembles) with
# `epochs: {start: 9, step: 10}`, i.e. after epochs 10, 20, ...: 3 times in a
# 32-epoch run, 12 in a 128-epoch one.
HEAVY_INFERENCE_EVERY = 10
HEAVY_INFERENCE_RESIDUE = 0


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


def progress(named, name: str) -> tuple[int, Seconds, float]:
    """(epochs done, median seconds per (epoch, training, light and heavy
    inference), useful train seconds) over the logs of every run (preemption
    retry) of the experiment, finished or not. The ft runs restart the epoch
    count, so epochs done are their own (0..max_epochs).
    """
    none: Seconds = (None, None, None, None)
    max_epochs = max_epochs_of(name)
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
            intervals, epoch_intervals(log, HEAVY_INFERENCE_RESIDUE)
        ):
            acc.extend(values)
    if named.status == OK:
        done = max_epochs
    full, train, light, heavy = intervals
    medians = (_median(full), _median(train), _median(light), _median(heavy))
    return done, medians, useful


def main() -> None:
    names = list(FT_RUNS)
    # One listing per distinct common prefix of a row's job names (the ext80
    # parent's name up to the seed, shared by an architecture's two rows): the
    # broad "ace2-fm-nc-" listing matches ~9k old experiments and takes
    # ~5 min, while a narrow one takes a few seconds.
    prefixes = dict.fromkeys(
        os.path.commonprefix([n for n in names if arch_of(n) == arch]) for arch in ARCHS
    )
    by_name = {}
    for prefix in prefixes:
        by_name.update(newest_by_name(prefix))
    with ThreadPoolExecutor(20) as ex:
        prog = dict(
            zip(
                names,
                ex.map(lambda n: progress(by_name.get(n), n), names),
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
        # Epoch columns cover unfinished (running or queued) jobs only.
        eps = [prog[n][0] for n, s in live_pairs if s == "running"]
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
                min(eps) if eps else "-",
                max(eps) if eps else "-",
                f"{statistics.median(eps):g}" if eps else "-",
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
