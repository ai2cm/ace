"""Progress table for the FM training runs currently tracked.

Rows (ROWS), 3 seeds each unless noted:
- The 18 FM v3 per-grid-cell normalization runs (create_fm_v3_jobs.py, 120
  epochs, 4 GPUs), one row per network x loss normalization:
  sfno-v3-{c1,c2,c3}-{scalarloss,cellloss}.
- Their smoke tests (sfno-v3-smoke-{c2,c3}-cellloss, s0, 2 epochs), while
  their configs exist.
- The 6 FM C96 AMIP-only runs (create_fm_c96amip_jobs.py, 120 epochs, 4
  GPUs), one row per architecture: sfno-v2-c96amip, swin-v2.2-v2-c96amip;
  and their smoke tests (sfno-v2-c96amip-smoke, swin-v2.2-v2-c96amip-smoke,
  s0, 2 epochs), while their configs exist.
- The 6 1deg, 6-hourly FM runs of create_fm_1deg_jobs (120 epochs, 8 GPUs),
  one row per arm: sfno-v2-1deg-fm-e3sm-balanced, sfno-v2-1deg-fm-era5.

Two backends, chosen by whether the `beaker` CLI is on PATH:
- beaker: one `beaker workspace experiments` listing per row prefix
  (_beaker_listing) and `beaker job logs` of every job (preemption retry) of
  each experiment, as before.
- gantry (agent VMs, which have no beaker CLI): the gantry shim
  (/usr/local/bin/gantry), which sees only jobs submitted from this VM. Job
  name -> Beaker id comes from records/fm_v3_jobs.json and
  records/fm_c96amip_jobs.json (written by submit_fm_v3_jobs.py and
  submit_fm_c96amip_jobs.py), else from `gantry list --all` (experiment names are
  agent-alexeyy-<job name>-<suffix>; the newest wins); status from `gantry
  list --all` (a job it calls running counts as queued until `gantry status`
  gives it a start time). `gantry logs <id>` holds only the experiment's
  CURRENT attempt (a resumed attempt starts at fme's "Resuming training from"
  line), so per-epoch timings, train time and restarts come from W&B instead
  (read-only wandb.Api, project ai2cm/FM): the run whose config.experiment_dir
  is the experiment's $AGENT_OUTPUT_DIR, which every attempt resumes. Its
  history has one row per completed epoch, across all attempts, with fme's
  epoch_{total,train,inference}_seconds. "Preemptions" counts the restarts
  W&B shows (a jump of more than RESTART_GAP_SECONDS between the wall-clock
  _timestamp and the run's own _runtime from one epoch to the next), plus 1
  if the current attempt has resumed and not finished an epoch yet (its log
  starts with "Resuming training from"); a restart counts whatever its cause.
  `gantry status` gives no start time to a preempted experiment waiting for
  its retry either, so a queued job whose W&B run shows progress is counted
  as queued and listed under the tables as "queued (preempted, epoch N)",
  not as never started. Without a W&B run (not started, no epoch done yet,
  or W&B unreachable) a job falls back to its current-attempt log. Rows of
  runs submitted outside this VM (the 1deg runs) are left out, with a note.

Beaker backend (gantry backend: see above for W&B): epochs done and
per-phase timings come from the logs of every run (preemption retry) of each
finished or running experiment. Epochs done is the
highest N in "Starting training step for model trained for N complete
epochs", so an epoch counts only once its training, validation and inline
inference are all done. Finished jobs count as their max_epochs; never-started
jobs count as 0.
Train time is the wall time that produced kept progress,
summed over an experiment's runs (preemption retries): each run counts from
its first training step to its exit (or now, if running), except that a
preempted or failed run counts only up to its last checkpoint save, since the
work after it is redone by the next retry. Queueing, startup and lost work are
excluded, so preemptions do not inflate it. (gantry backend: the sum of
W&B's epoch_total_seconds over the completed epochs, i.e. the same kept
work, minus the epoch in progress.) Epoch columns cover unfinished (running or
queued) jobs only, "-" when none; train-time and timing columns cover finished
and running jobs. The "inference, every 10" column is the inline inference
after epochs 10, 20, ..., when the every-10th-epoch entries run too; "-" until
a run of the row has finished one (the smoke test runs all of its inference
every epoch). All values are measured; there are no estimates.

"Not submitted" counts job names with no Beaker experiment;
"Failed / Other" counts only experiments that exist and failed or were canceled.
"Preemptions" sums, over the row's experiments, the jobs Beaker canceled
because a higher-priority job preempted them (gantry backend: the restarts).

Usage (from the repo root):
    PYTHONPATH=$PWD python configs/experiments/2026-10-19-fm/scripts/fm_table.py
"""

import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from create_fm_1deg_jobs import ARMS as ARMS_1DEG  # noqa: E402
from create_fm_1deg_jobs import SEEDS as SEEDS_1DEG  # noqa: E402
from create_fm_1deg_jobs import run_config_name as run_config_name_1deg  # noqa: E402
from create_fm_c96amip_jobs import ARCHS as C96AMIP_ARCHS  # noqa: E402
from create_fm_c96amip_jobs import MAX_EPOCHS as MAX_EPOCHS_C96AMIP  # noqa: E402
from create_fm_c96amip_jobs import (
    SMOKE_MAX_EPOCHS as C96AMIP_SMOKE_EPOCHS,  # noqa: E402
)
from create_fm_c96amip_jobs import runs as c96amip_runs  # noqa: E402
from create_fm_c96amip_jobs import smoke_runs as c96amip_smoke_runs  # noqa: E402
from create_fm_v3_jobs import LOSSES, NETWORKS, SMOKE_MAX_EPOCHS  # noqa: E402
from create_fm_v3_jobs import MAX_EPOCHS as MAX_EPOCHS_V3  # noqa: E402
from create_fm_v3_jobs import runs as v3_runs  # noqa: E402
from create_fm_v3_jobs import smoke_runs as v3_smoke_runs  # noqa: E402
from submit_fm_jobs import config_to_job_name  # noqa: E402

EXP_DIR = pathlib.Path(__file__).parent.parent
V3_RECORD = EXP_DIR / "records" / "fm_v3_jobs.json"
C96AMIP_RECORD = EXP_DIR / "records" / "fm_c96amip_jobs.json"
# The submit scripts' records, each naming the latest submission of its jobs.
RECORDS = (V3_RECORD, C96AMIP_RECORD)
GANTRY = "/usr/local/bin/gantry"
AGENT_PREFIX = "agent-alexeyy-"
GANTRY_LIST_LIMIT = 500
# W&B (gantry backend): the runs' project, and where the jobs write
# ($AGENT_OUTPUT_DIR = AGENT_OUTPUT_ROOT/<experiment>, the runs'
# config.experiment_dir).
WANDB_PATH = "ai2cm/FM"
AGENT_OUTPUT_ROOT = "/climate-default/agents/alexeyy"
WANDB_HISTORY_KEYS = (
    "epoch",
    "_timestamp",
    "_runtime",
    "epoch_total_seconds",
    "epoch_train_seconds",
    "epoch_inference_seconds",
)
# Between two epochs of one W&B run, wall time beyond the run's own runtime:
# a restart (the run was down while Beaker requeued the job). Within an
# attempt the two agree to ~2 minutes.
RESTART_GAP_SECONDS = 600

# "Starting training step for model trained for N complete epochs" marks that
# epochs 0..N-1 are done, including their validation and inline inference.
EPOCH_RE = re.compile(r"Starting training step for model trained for (\d+) complete")
# Phase starts per epoch. A full epoch runs from one training start to the
# next; training from a training start to the next validation start; inline
# inference from an inference start to the next training start (it includes
# the checkpoint save). A resumed run's first training start is mid-epoch, so
# its epoch and training intervals are skipped.
PHASE_RE = re.compile(
    r"Starting (training|validation|inference) step for model trained for (\d+)"
)
RESUME_MARK = "Resuming training from"
# Checkpoint saves: at each epoch end and every checkpoint_every_n_batches. A
# preempted run resumes from the last one, so its later work is redone.
CHECKPOINT_RE = re.compile(r"Saving (?:checkpoints for epoch|latest checkpoint)")
# Line timestamps: the RFC 3339 prefix `beaker job logs` adds, else fme's own
# logging asctime (UTC in the job containers).
BEAKER_TIME_RE = re.compile(r"^(\S+)Z ")
ASCTIME_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) - ")
LOG_FILTER_RE = re.compile(
    r"Starting (training|validation|inference) step|Resuming training"
    r"|Saving (checkpoints for epoch|latest checkpoint)"
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


# --- tracked runs ----------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    label: str
    names: tuple[str, ...]
    max_epochs: int
    # Submitted from this agent VM (visible through the gantry shim).
    agent_vm: bool


def _rows() -> list[Row]:
    rows = []
    v3 = v3_runs()
    for network in NETWORKS:
        for loss in LOSSES:
            names = tuple(
                r.job_name for r in v3 if (r.network, r.loss) == (network, loss)
            )
            rows.append(Row(f"sfno-v3-{network}-{loss}", names, MAX_EPOCHS_V3, True))
    for smoke in v3_smoke_runs():
        if smoke.path.exists():
            label = f"sfno-v3-smoke-{smoke.network}-{smoke.loss}"
            rows.append(Row(label, (smoke.job_name,), SMOKE_MAX_EPOCHS, True))
    c96amip = c96amip_runs()
    for arch in C96AMIP_ARCHS:
        names = tuple(r.job_name for r in c96amip if r.arch == arch)
        label = f"{arch.removeprefix('nc-')}-c96amip"
        rows.append(Row(label, names, MAX_EPOCHS_C96AMIP, True))
    for c96amip_smoke in c96amip_smoke_runs():
        if c96amip_smoke.path.exists():
            label = f"{c96amip_smoke.arch.removeprefix('nc-')}-c96amip-smoke"
            names = (c96amip_smoke.job_name,)
            rows.append(Row(label, names, C96AMIP_SMOKE_EPOCHS, True))
    for arm in ARMS_1DEG:
        names = tuple(
            config_to_job_name(run_config_name_1deg(arm, seed, False))
            for seed in SEEDS_1DEG
        )
        rows.append(Row(f"sfno-v2-1deg-{arm}", names, 120, False))
    return rows


ROWS = _rows()


@dataclass
class Tracked:
    """One job name's experiment, from either backend."""

    status: str  # OK, "running", or how it ended
    queued: bool
    preemptions: int
    # Queued after progress (preempted, waiting for its retry): epochs done.
    preempted_at_epoch: int | None = None


Seconds = tuple[float | None, float | None, float | None, float | None]
Intervals = tuple[list[float], list[float], list[float], list[float]]
Progress = tuple[int, Seconds, float]
NO_SECONDS: Seconds = (None, None, None, None)


# --- log parsing (both backends) -------------------------------------------------


def _line_time(line: str) -> datetime | None:
    match = BEAKER_TIME_RE.match(line)
    if match:
        return datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC)
    match = ASCTIME_RE.search(line)
    if match:
        t = datetime.fromisoformat(f"{match.group(1)}.{match.group(2)}")
        return t.replace(tzinfo=UTC)
    return None


def _events(log: str) -> list[tuple[datetime, str, int]]:
    events = []
    for line in log.splitlines():
        match = PHASE_RE.search(line)
        t = _line_time(line) if match else None
        if match and t is not None:
            events.append((t, match.group(1), int(match.group(2))))
    return events


def _checkpoint_times(log: str) -> list[datetime]:
    return [
        t
        for line in log.splitlines()
        if CHECKPOINT_RE.search(line) and (t := _line_time(line)) is not None
    ]


def _last_time(log: str) -> datetime | None:
    for line in reversed(log.splitlines()):
        if (t := _line_time(line)) is not None:
            return t
    return None


def _kept_seconds(log: str, end_kind: str, exited: datetime | None) -> float:
    """Wall seconds of one run whose progress was kept: from its first
    training step to `exited` ("ok"), now ("running"), or its last checkpoint
    save ("lost": preempted or failed, as later work is redone).
    """
    starts = [t for t, _, _ in _events(log)]
    if not starts:
        return 0.0
    start = starts[0]
    if end_kind == "ok" and exited is not None:
        end = exited
    elif end_kind == "running":
        end = datetime.now(UTC)
    else:
        saves = [t for t in _checkpoint_times(log) if t >= start]
        end = saves[-1] if saves else start
    return max((end - start).total_seconds(), 0.0)


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 1440}d {minutes // 60 % 24}h {minutes % 60}m"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


# Inline inference runs the aimip evaluator every epoch and all others (the
# 10yr and 46yr ERA5 runs and the 5-day weather ensembles) with
# `epochs: {start: 9, step: 10}`, i.e. after epochs 10, 20, ...: 12 times in a
# 120-epoch run.
HEAVY_INFERENCE_EVERY = 10
HEAVY_INFERENCE_RESIDUE = 0


def _is_heavy_inference(epoch: int, residue: int) -> bool:
    return epoch % HEAVY_INFERENCE_EVERY == residue


def epoch_intervals(log: str, heavy_residue: int) -> Intervals:
    """Wall seconds of each (full epoch, training, inline inference on light
    epochs, inline inference on heavy every-10th epochs) in one run's log.
    """
    events = _events(log)
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


def _progress_from_runs(
    runs: list[tuple[str, str, datetime | None]], finished: bool, max_epochs: int
) -> Progress:
    """(epochs done, median seconds per (epoch, training, light and heavy
    inference), kept train seconds) over the logs of every run (preemption
    retry) of an experiment: (log, end kind, exit time) each.
    """
    if not runs:
        return (max_epochs if finished else 0), NO_SECONDS, 0.0
    intervals: Intervals = ([], [], [], [])
    done = 0
    kept = 0.0
    for log, end_kind, exited in runs:
        kept += _kept_seconds(log, end_kind, exited)
        done = max(done, *(int(m.group(1)) for m in EPOCH_RE.finditer(log)), 0)
        for acc, values in zip(
            intervals, epoch_intervals(log, HEAVY_INFERENCE_RESIDUE)
        ):
            acc.extend(values)
    if finished:
        done = max_epochs
    full, train, light, heavy = intervals
    return done, (_median(full), _median(train), _median(light), _median(heavy)), kept


def _filter_log(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if LOG_FILTER_RE.search(line))


# --- beaker backend ------------------------------------------------------------


def _submitted(experiment: dict) -> str:
    """When the experiment was submitted: its first job's creation time.

    Not the latest job's, which moves forward on every preemption retry.
    """
    jobs = experiment.get("jobs") or []
    created = [job.get("status", {}).get("created", "") for job in jobs]
    return min(created, default="")


def newest_by_name(text: str, names: set[str]) -> dict[str, NamedExperiment]:
    """The newest experiment for each canonical job name matching ``text``.

    Unlike ``_beaker_listing.experiments_by_name`` (which prefers the best
    status, as the submit scripts need), a resubmitted job always replaces the
    older experiment of the same name, so the table tracks the latest run.
    Experiments submitted through the agent VM's proxy
    (agent-alexeyy-<job name>-<suffix>) are matched to the tracked `names`.
    """
    newest: dict[str, dict] = {}
    for experiment in list_experiments(text=text):
        if experiment["name"].startswith(AGENT_PREFIX):
            name = _job_name_of(experiment["name"], names)
            if name is None:
                continue
        else:
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


def _beaker_job_log(job_id: str) -> str:
    proc = subprocess.run(
        ["beaker", "job", "logs", job_id], capture_output=True, text=True
    )
    return _filter_log(proc.stdout)


def _beaker_progress(named: NamedExperiment | None, max_epochs: int) -> Progress:
    if named is None:
        return 0, NO_SECONDS, 0.0
    jobs = named.experiment.get("jobs") or []
    runs = []
    for job in jobs:
        status = job.get("status", {})
        if "started" not in status:
            continue
        if status.get("exited") and status.get("exitCode") == 0:
            kind, exited = "ok", datetime.fromisoformat(status["exited"])
        elif not status.get("exited") and not status.get("canceled"):
            kind, exited = "running", None
        else:
            kind, exited = "lost", None
        runs.append((_beaker_job_log(job["id"]), kind, exited))
    return _progress_from_runs(runs, named.status == OK, max_epochs)


def beaker_backend(rows: list[Row]) -> tuple[dict[str, Tracked], dict[str, Progress]]:
    # One listing per distinct common prefix of a row's job names (the name up
    # to the seed; it also matches the row's smoke tests, which are dropped
    # as their names are not tracked): the broad "ace2-fm-nc-" listing matches
    # ~9k old experiments and takes ~5 min, while a narrow one takes seconds.
    prefixes = dict.fromkeys(os.path.commonprefix(list(row.names)) for row in rows)
    max_epochs = {name: row.max_epochs for row in rows for name in row.names}
    by_name: dict[str, NamedExperiment] = {}
    for prefix in prefixes:
        by_name.update(newest_by_name(prefix, set(max_epochs)))
    names = list(max_epochs)
    with ThreadPoolExecutor(20) as ex:
        progress = dict(
            zip(
                names,
                ex.map(
                    lambda n: _beaker_progress(by_name.get(n), max_epochs[n]), names
                ),
            )
        )
    tracked = {
        name: Tracked(named.status, _queued(named), _preemptions(named))
        for name, named in by_name.items()
        if name in max_epochs
    }
    return tracked, progress


# --- gantry backend --------------------------------------------------------------

# gantry list statuses -> the beaker backend's statuses (anything else counts
# as Failed / Other).
GANTRY_DONE = {"succeeded", "success", "completed", "finished", "ok", "done"}
GANTRY_RUNNING = {"running", "started", "initializing", "executing"}
GANTRY_QUEUED = {"queued", "pending", "submitted", "scheduled", "created", "waiting"}
GANTRY_LIST_RE = re.compile(r"^(?P<id>\S+)\s+(?P<status>\S+)\s+(?P<rest>.*)$")


@dataclass
class GantryJob:
    id: str
    status: str
    created: str
    name: str


def _gantry(*args: str) -> str:
    proc = subprocess.run([GANTRY, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gantry {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _gantry_list() -> list[GantryJob]:
    """`gantry list --all`: `<id>  <status>  <created>  <name>` per line."""
    jobs = []
    out = _gantry("list", "--all", "--limit", str(GANTRY_LIST_LIMIT))
    for line in out.splitlines():
        match = GANTRY_LIST_RE.match(line.strip())
        if match is None or line.startswith("no jobs"):
            continue
        rest = match["rest"].split()
        name = rest[-1]
        created = rest[0] if len(rest) > 1 else ""
        jobs.append(GantryJob(match["id"], match["status"].lower(), created, name))
    return jobs


def _gantry_started(job: GantryJob) -> bool:
    """Whether `gantry status` gives the job a start time."""
    return bool(json.loads(_gantry("status", job.id)).get("started"))


def _job_name_of(experiment: str, names: set[str]) -> str | None:
    """The tracked job name of an experiment agent-alexeyy-<name>-<suffix>."""
    if not experiment.startswith(AGENT_PREFIX):
        return None
    body = experiment.removeprefix(AGENT_PREFIX)
    if body in names:
        return body
    stem, sep, suffix = body.rpartition("-")
    return stem if sep and stem in names and suffix else None


def _gantry_status(status: str) -> tuple[str, bool]:
    if status in GANTRY_DONE:
        return OK, False
    if status in GANTRY_RUNNING:
        return "running", False
    if status in GANTRY_QUEUED:
        return "running", True
    return status or "unknown", False


def _split_runs(log: str) -> list[str]:
    """One experiment's log split into its runs (preemption retries), each
    resumed run starting at its "Resuming training from" line.
    """
    runs: list[list[str]] = [[]]
    for line in log.splitlines():
        if RESUME_MARK in line and runs[-1]:
            runs.append([])
        runs[-1].append(line)
    return ["\n".join(run) for run in runs if run]


def _gantry_progress(
    job: GantryJob | None, max_epochs: int
) -> tuple[Progress, int, bool]:
    """(progress, restarts, pending restart) of one experiment from the log of
    its current attempt (`gantry logs`); pending: the attempt resumed and has
    not finished an epoch since.
    """
    if job is None:
        return (0, NO_SECONDS, 0.0), 0, False
    status, queued = _gantry_status(job.status)
    if queued:
        return (0, NO_SECONDS, 0.0), 0, False
    log = _filter_log(_gantry("logs", job.id))
    runs = _split_runs(log)
    entries = []
    for i, run_log in enumerate(runs):
        last = i == len(runs) - 1
        if last and status == "running":
            kind, exited = "running", None
        elif last and status == OK:
            kind, exited = "ok", _last_time(run_log)
        else:
            kind, exited = "lost", None
        entries.append((run_log, kind, exited))
    restarts = sum(RESUME_MARK in run for run in runs)
    epochs = {int(m.group(1)) for m in EPOCH_RE.finditer(runs[-1])} if runs else set()
    pending = bool(runs) and RESUME_MARK in runs[-1] and len(epochs) <= 1
    progress = _progress_from_runs(entries, status == OK, max_epochs)
    return progress, restarts, pending


def _wandb_histories(experiments: dict[str, str]) -> dict[str, list[dict]]:
    """Job name -> per-epoch W&B history rows (WANDB_HISTORY_KEYS, sorted by
    epoch, last row per epoch) of the run of each experiment (job name ->
    experiment name) that has completed an epoch. {} if W&B is unreachable.
    """
    if not experiments:
        return {}
    try:
        import wandb

        api = wandb.Api(timeout=60)
        dirs = {f"{AGENT_OUTPUT_ROOT}/{exp}": name for name, exp in experiments.items()}
        found: dict[str, object] = {}
        filters = {"config.experiment_dir": {"$in": list(dirs)}}
        for run in api.runs(WANDB_PATH, filters=filters, per_page=100):
            name = dirs.get(run.config.get("experiment_dir"))
            if name is not None:
                found[name] = run

        def history(run) -> list[dict]:
            rows = run.history(
                keys=list(WANDB_HISTORY_KEYS), samples=10000, pandas=False
            )
            by_epoch = {int(row["epoch"]): row for row in rows}
            return [by_epoch[e] for e in sorted(by_epoch)]

        with ThreadPoolExecutor(13) as ex:
            histories = dict(zip(found, ex.map(history, found.values())))
    except Exception as err:  # noqa: BLE001 (any W&B failure: fall back to logs)
        print(f"W&B unavailable ({err!r}); timings from current-attempt logs only.")
        return {}
    return {name: rows for name, rows in histories.items() if rows}


def _wandb_progress(
    rows: list[dict], finished: bool, max_epochs: int
) -> tuple[Progress, int]:
    """(progress, restarts seen) from one run's per-epoch W&B history."""
    full = [float(r["epoch_total_seconds"]) for r in rows]
    train = [float(r["epoch_train_seconds"]) for r in rows]
    light: list[float] = []
    heavy: list[float] = []
    for r in rows:
        heavy_epoch = _is_heavy_inference(int(r["epoch"]), HEAVY_INFERENCE_RESIDUE)
        (heavy if heavy_epoch else light).append(float(r["epoch_inference_seconds"]))
    restarts = sum(
        (b["_timestamp"] - a["_timestamp"]) - (b["_runtime"] - a["_runtime"])
        > RESTART_GAP_SECONDS
        for a, b in zip(rows, rows[1:])
    )
    done = max_epochs if finished else int(rows[-1]["epoch"])
    seconds = (_median(full), _median(train), _median(light), _median(heavy))
    return (done, seconds, sum(full)), restarts


def gantry_backend(rows: list[Row]) -> tuple[dict[str, Tracked], dict[str, Progress]]:
    max_epochs = {name: row.max_epochs for row in rows for name in row.names}
    names = set(max_epochs)
    listed = _gantry_list()
    by_id = {job.id: job for job in listed}
    newest: dict[str, GantryJob] = {}
    for job in listed:
        name = _job_name_of(job.name, names)
        if name is not None and (
            name not in newest or job.created > newest[name].created
        ):
            newest[name] = job
    # The submit script's record names the latest submission of each job.
    for record in RECORDS:
        if not record.exists():
            continue
        for name, entry in json.loads(record.read_text()).items():
            if name in names and entry.get("id") in by_id:
                newest[name] = by_id[entry["id"]]
    # `gantry list` can report a job as running before its container has
    # started (it then shows up as queued again); only a job with a start time
    # in `gantry status` counts as running.
    running = [job for job in newest.values() if job.status in GANTRY_RUNNING]
    with ThreadPoolExecutor(13) as ex:
        for job, started in zip(running, ex.map(_gantry_started, running)):
            if not started:
                job.status = "queued"
    ordered = list(max_epochs)
    with ThreadPoolExecutor(13) as ex:
        results = dict(
            zip(
                ordered,
                ex.map(
                    lambda n: _gantry_progress(newest.get(n), max_epochs[n]), ordered
                ),
            )
        )
    histories = _wandb_histories({name: job.name for name, job in newest.items()})
    tracked = {}
    progress = {name: result[0] for name, result in results.items()}
    for name, job in newest.items():
        status, queued = _gantry_status(job.status)
        log_progress, log_restarts, pending = results[name]
        if name not in histories:
            tracked[name] = Tracked(status, queued, log_restarts)
            continue
        progress[name], restarts = _wandb_progress(
            histories[name], status == OK, max_epochs[name]
        )
        # The log's epoch count is ahead while an epoch is being logged.
        if not queued and status != OK:
            done, seconds, kept = progress[name]
            progress[name] = (max(done, log_progress[0]), seconds, kept)
        preempted_at = progress[name][0] if queued else None
        tracked[name] = Tracked(
            status, queued, restarts + int(pending), preempted_at_epoch=preempted_at
        )
    return tracked, progress


# --- table -----------------------------------------------------------------------


def main() -> None:
    use_beaker = shutil.which("beaker") is not None
    rows = ROWS if use_beaker else [row for row in ROWS if row.agent_vm]
    if use_beaker:
        tracked, prog = beaker_backend(rows)
    else:
        tracked, prog = gantry_backend(rows)
    count_rows = []
    timing_rows = []
    for row in rows:
        statuses = [tracked[n].status if n in tracked else "missing" for n in row.names]
        # Epoch columns cover finished and running jobs only; failed/canceled
        # or deleted jobs would pull them toward 0.
        live_pairs = [
            (n, s) for n, s in zip(row.names, statuses) if s in (OK, "running")
        ]
        live = [n for n, _ in live_pairs]
        # Epoch columns cover unfinished (running or queued) jobs only.
        eps = [prog[n][0] for n, s in live_pairs if s == "running"]
        # Train time covers finished and running jobs; a queued job has none
        # yet and would pull the median toward 0.
        times = [prog[n][2] for n in live if not tracked[n].queued]

        def median_column(i: int) -> str:
            values = [s for n in live if (s := prog[n][1][i]) is not None]
            return f"{statistics.median(values):.0f}" if values else "-"

        count_rows.append(
            (
                row.label,
                sum(s == OK for s in statuses),
                sum(s == "running" and not tracked[n].queued for n, s in live_pairs),
                sum(s == "running" and tracked[n].queued for n, s in live_pairs),
                sum(s not in (OK, "running", "missing") for s in statuses),
                sum(s == "missing" for s in statuses),
                sum(tracked[n].preemptions for n in row.names if n in tracked),
                min(eps) if eps else "-",
                max(eps) if eps else "-",
                f"{statistics.median(eps):g}" if eps else "-",
            )
        )
        timing_rows.append(
            (
                row.label,
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
    preempted = [
        f"{n} (epoch {t.preempted_at_epoch})"
        for row in rows
        for n in row.names
        if (t := tracked.get(n)) is not None and t.preempted_at_epoch is not None
    ]
    if preempted:
        print()
        print("Queued (preempted, waiting for a retry): " + ", ".join(preempted) + ".")
    if not use_beaker:
        hidden = [row.label for row in ROWS if not row.agent_vm]
        print()
        print(
            "Source: gantry shim (no beaker CLI here; it sees only jobs "
            "submitted from this VM) and W&B (timings, train time, restarts). "
            "Not shown: " + ", ".join(hidden) + "."
        )


def _print_table(columns: tuple[str, ...], rows: list[tuple]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")


if __name__ == "__main__":
    main()
