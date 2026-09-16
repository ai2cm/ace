"""Beaker as the source of truth for which FM jobs exist and how they ended.

Every submit script and the job watcher ask the same three questions of the
`ai2/ace` workspace: which experiments carry an FM job name, did each one
succeed, and which result dataset holds a succeeded training run's
checkpoints. This module answers them from one `beaker workspace experiments`
listing rather than from wandb, whose run listing takes minutes and lags the
job by however long wandb takes to sync.

Gantry appends a short hash to a job name when the name is already taken in
the workspace (`ace2-fm-...-p0k` becomes `ace2-fm-...-p0k-c2e7`), so an
experiment is matched to a job name by its *canonical* name, with any such
suffix stripped. See `canonical_name`.
"""

import json
import re
import subprocess
from dataclasses import dataclass

DEFAULT_WORKSPACE = "ai2/ace"
DEFAULT_TEXT = "ace2-fm-"

# Gantry's collision suffix: a dash and four lowercase hex digits at the very
# end of the name. An epoch segment such as `-e07` is three characters and
# never matches, and no FM job-name segment is itself four hex digits.
_HASH_SUFFIX = re.compile(r"-[0-9a-f]{4}$")

# How a job ended. Beaker sets `canceled`/`canceledCode` both on a job a user
# stopped and on one the scheduler preempted; a preempted job also carries a
# non-zero exit code, so the exit code alone cannot tell a crash from a
# preemption. Preemption is recognized from `canceledFor`, and the scheduler
# retries a preempted job as a new job in the same experiment. An experiment
# whose latest job was preempted with no retry queued is treated as failed:
# something has to resubmit it.
OK = "ok"
RUNNING = "running"
FAILED = "failed"
CANCELED = "canceled"
NO_JOB = "nojob"

_PREEMPTED_MARKER = "preempted"


def canonical_name(name: str) -> str:
    return _HASH_SUFFIX.sub("", name)


def job_status(job: dict) -> str:
    status = job.get("status", {})
    if "exited" not in status and "canceled" not in status:
        return RUNNING
    if status.get("exitCode") == 0:
        return OK
    if "canceled" in status or status.get("canceledCode") is not None:
        if _PREEMPTED_MARKER in (status.get("canceledFor") or ""):
            return FAILED
        return CANCELED
    return FAILED


def experiment_status(experiment: dict) -> str:
    """One status for an experiment: ok if any job succeeded, else running if
    one is still going, else the way the latest job ended.
    """
    statuses = [job_status(job) for job in experiment.get("jobs") or []]
    if not statuses:
        return NO_JOB
    if OK in statuses:
        return OK
    if RUNNING in statuses:
        return RUNNING
    return statuses[-1]


def succeeded_result_dataset(experiment: dict) -> str | None:
    """Result dataset of the latest exit-0 job, or None."""
    succeeded = [job for job in experiment.get("jobs") or [] if job_status(job) == OK]
    if not succeeded:
        return None
    succeeded.sort(key=lambda job: job.get("status", {}).get("started", ""))
    return succeeded[-1].get("result", {}).get("beaker")


def list_experiments(
    workspace: str = DEFAULT_WORKSPACE, text: str = DEFAULT_TEXT
) -> list[dict]:
    proc = subprocess.run(
        [
            "beaker",
            "workspace",
            "experiments",
            workspace,
            "--text",
            text,
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


@dataclass(frozen=True)
class NamedExperiment:
    """The experiment standing for one canonical job name.

    When several experiments share a canonical name (a plain resubmission and
    a hash-suffixed one), the one that succeeded wins, then one still running,
    then the most recently created.
    """

    name: str
    status: str
    experiment: dict

    @property
    def id(self) -> str:
        return self.experiment["id"]

    @property
    def result_dataset(self) -> str | None:
        return succeeded_result_dataset(self.experiment)


_RANK = {OK: 0, RUNNING: 1, FAILED: 2, CANCELED: 3, NO_JOB: 4}


def _created(experiment: dict) -> str:
    """Latest job creation time; ISO-8601 strings from beaker sort as text."""
    jobs = experiment.get("jobs") or []
    return max((job.get("status", {}).get("created", "") for job in jobs), default="")


def experiments_by_name(experiments: list[dict]) -> dict[str, NamedExperiment]:
    grouped: dict[str, list[dict]] = {}
    for experiment in experiments:
        grouped.setdefault(canonical_name(experiment["name"]), []).append(experiment)
    best: dict[str, NamedExperiment] = {}
    for name, group in grouped.items():
        # Newest first, so that min() over the status rank keeps the most
        # recent experiment among those tied on status.
        group.sort(key=_created, reverse=True)
        experiment = min(group, key=lambda e: _RANK[experiment_status(e)])
        best[name] = NamedExperiment(name, experiment_status(experiment), experiment)
    return best


def fetch_experiments_by_name(
    workspace: str = DEFAULT_WORKSPACE, text: str = DEFAULT_TEXT
) -> dict[str, NamedExperiment]:
    return experiments_by_name(list_experiments(workspace, text))


def existing_job_names(
    workspace: str = DEFAULT_WORKSPACE, text: str = DEFAULT_TEXT
) -> set[str]:
    """Canonical job names that succeeded or are still running.

    This is what `--skip-if-in-beaker` skips: a failed or canceled experiment
    leaves its name free to be resubmitted.
    """
    return {
        name
        for name, named in fetch_experiments_by_name(workspace, text).items()
        if named.status in (OK, RUNNING)
    }
