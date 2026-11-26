"""Beaker as seen from an agent VM, for the status tables and submit scripts.

On an agent VM (recognized by AGENT_VM_MARKER, the Beaker proxy's config) there
is no real `beaker` CLI: `beaker` is a read-only shim whose
`workspace experiments` listing is capped at BEAKER_LIST_LIMIT experiments,
gives one experiment-level status per experiment (no jobs) and has no
`job logs`; `gantry` (GANTRY) is a shim that submits through the proxy and
lists, with `gantry list --all`, only the jobs submitted through it. The
proxy names an experiment agent-alexeyy-<job name>-<suffix> (AGENT_PREFIX).

This module answers, through those shims, what _beaker_listing answers with
the real CLI: the status of a recorded experiment id (`gantry status`, else
`beaker experiment get` for an experiment gantry does not know), and which
experiments carry given job names (`gantry list --all`, the submit scripts'
records, and one capped `beaker workspace experiments --text` listing per
group of names, which also sees experiments submitted elsewhere, e.g. with
the real gantry and no prefix).

Statuses are the experiment-level ones the proxy reports (for a preempted
and retried experiment, that of its latest job): GANTRY_DONE, GANTRY_RUNNING
and GANTRY_QUEUED map to _beaker_listing's OK and RUNNING (gantry_status).
"""

import json
import pathlib
import subprocess
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from _beaker_listing import CANCELED, DEFAULT_WORKSPACE, FAILED, OK, RUNNING
from _beaker_listing import canonical_name as _canonical_name

GANTRY = "/usr/local/bin/gantry"
BEAKER = "/usr/local/bin/beaker"
# Present on agent VMs only (the Beaker proxy's config, read by the gantry and
# beaker shims).
AGENT_VM_MARKER = pathlib.Path("/etc/agent-sandbox/proxy.toml")
AGENT_PREFIX = "agent-alexeyy-"
# Beaker rejects experiment names over 128 characters; the proxy adds
# AGENT_PREFIX and a "-<4 characters>" suffix to the --name it is given.
AGENT_SUFFIX_LENGTH = 4
AGENT_NAME_LIMIT = 128 - len(AGENT_PREFIX) - 1 - AGENT_SUFFIX_LENGTH
GANTRY_LIST_LIMIT = 5000
# The beaker shim's cap on `beaker workspace experiments --limit`.
BEAKER_LIST_LIMIT = 100
LOOKUP_THREADS = 8

# gantry / beaker shim statuses -> _beaker_listing's (anything else counts as
# failed or canceled, gantry_status).
GANTRY_DONE = {"succeeded", "success", "completed", "finished", "ok", "done"}
GANTRY_RUNNING = {"running", "started", "initializing", "executing"}
GANTRY_QUEUED = {"queued", "pending", "submitted", "scheduled", "created", "waiting"}
GANTRY_CANCELED = {"canceled", "cancelled", "stopped"}


def on_agent_vm() -> bool:
    return AGENT_VM_MARKER.exists()


@dataclass
class GantryJob:
    """One experiment as the shims report it (no per-job detail)."""

    id: str
    status: str
    created: str
    name: str
    exit_code: int | None = None


def gantry(*args: str) -> str:
    proc = subprocess.run([GANTRY, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gantry {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def gantry_list() -> list[GantryJob]:
    """`gantry list --all`: `<id>  <status>  [<priority>]  <created>  <name>`
    per line.
    """
    jobs = []
    out = gantry("list", "--all", "--limit", str(GANTRY_LIST_LIMIT))
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or line.startswith("no jobs"):
            continue
        rest = parts[2:]
        created = rest[-2] if len(rest) > 1 else ""
        jobs.append(GantryJob(parts[0], parts[1].lower(), created, rest[-1]))
    return jobs


def _from_status(status: dict) -> GantryJob:
    return GantryJob(
        status["id"],
        (status.get("status") or "").lower(),
        status.get("created") or "",
        status.get("name") or "",
        status.get("exit_code"),
    )


def gantry_job(job_id: str) -> GantryJob | None:
    """A job from `gantry status`; None if gantry does not know it."""
    try:
        return _from_status(json.loads(gantry("status", job_id)))
    except RuntimeError:
        return None


def gantry_started(job: GantryJob) -> bool:
    """Whether `gantry status` gives the job a start time."""
    return bool(json.loads(gantry("status", job.id)).get("started"))


def beaker_experiment(experiment_id: str) -> GantryJob | None:
    """An experiment from the beaker shim's `experiment get`; None if it
    cannot be read.
    """
    proc = subprocess.run(
        [BEAKER, "experiment", "get", experiment_id, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    experiment = json.loads(proc.stdout)
    jobs = [job for task in experiment.get("tasks") or [] for job in task["jobs"]]
    latest: dict = max(jobs, key=lambda job: job.get("created") or "", default={})
    return GantryJob(
        experiment["id"],
        (experiment.get("status") or "").lower(),
        experiment.get("created") or "",
        experiment.get("name") or "",
        latest.get("exit_code"),
    )


def experiment_job(experiment_id: str) -> GantryJob | None:
    """A recorded experiment: `gantry status`, else `beaker experiment get`."""
    return gantry_job(experiment_id) or beaker_experiment(experiment_id)


def gantry_status(status: str) -> tuple[str, bool]:
    """(_beaker_listing status, queued) of a shim status."""
    if status in GANTRY_DONE:
        return OK, False
    if status in GANTRY_RUNNING:
        return RUNNING, False
    if status in GANTRY_QUEUED:
        return RUNNING, True
    return status or "unknown", False


def training_status(job: GantryJob | None) -> str:
    """_beaker_listing status of a training experiment: OK only when its
    latest job (the experiment-level status) succeeded with exit code 0.
    """
    if job is None:
        return "not found"
    status, _ = gantry_status(job.status)
    if status == OK and job.exit_code not in (0, None):
        return FAILED
    if status in (OK, RUNNING):
        return status
    return CANCELED if job.status in GANTRY_CANCELED else FAILED


def job_name_of(experiment: str, names: Iterable[str]) -> str | None:
    """The job name among `names` of an experiment named
    agent-alexeyy-<name>-<suffix> (gantry shim) or <name>[-<hex>] (real
    gantry; _beaker_listing.canonical_name).
    """
    names = names if isinstance(names, set | frozenset | dict) else set(names)
    if not experiment.startswith(AGENT_PREFIX):
        name = _canonical_name(experiment)
        return name if name in names else None
    body = experiment.removeprefix(AGENT_PREFIX)
    if body in names:
        return body
    stem, sep, suffix = body.rpartition("-")
    return (
        stem if sep and stem in names and len(suffix) == AGENT_SUFFIX_LENGTH else None
    )


def beaker_listing(text: str, workspace: str = DEFAULT_WORKSPACE) -> list[GantryJob]:
    """The beaker shim's `workspace experiments --text text` (at most
    BEAKER_LIST_LIMIT experiments).
    """
    cmd = [BEAKER, "workspace", "experiments", workspace, "--text", text]
    cmd += ["--limit", str(BEAKER_LIST_LIMIT), "--format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"beaker listing of {text} failed: {proc.stderr.strip()}")
    return [
        GantryJob(
            e["id"], (e.get("status") or "").lower(), e.get("created", ""), e["name"]
        )
        for e in json.loads(proc.stdout)
    ]


def _complete_listing(text: str, names: set[str], workspace: str) -> list[GantryJob]:
    """beaker_listing of `text`, or, when it hits the cap (so may miss
    some of `names`), one listing per name, each of which must fit.
    """
    listed = beaker_listing(text, workspace)
    if len(listed) < BEAKER_LIST_LIMIT:
        return listed
    listed = []
    for name in sorted(names):
        by_name = beaker_listing(name, workspace)
        if len(by_name) >= BEAKER_LIST_LIMIT:
            raise RuntimeError(
                f"beaker listing of {name} hit the shim's cap of "
                f"{BEAKER_LIST_LIMIT}; cannot tell which experiments exist"
            )
        listed += by_name
    return listed


def find_experiments(
    names_by_text: Mapping[str, Iterable[str]],
    recorded: Mapping[str, str] | None = None,
    workspace: str = DEFAULT_WORKSPACE,
) -> dict[str, list[GantryJob]]:
    """Job name -> every experiment found for it, from `gantry list --all`,
    the `recorded` (job name -> experiment id) ids `gantry status` knows, and
    one beaker shim listing per key of `names_by_text` (each name must contain
    its key; a listing that hits the cap is redone name by name).
    """
    names = {name for group in names_by_text.values() for name in group}
    for text, group in names_by_text.items():
        assert all(text in name for name in group), (text, group)
    found: dict[str, dict[str, GantryJob]] = {}

    def add(job: GantryJob, name: str | None = None) -> None:
        name = name or job_name_of(job.name, names)
        if name is not None:
            found.setdefault(name, {})[job.id] = job

    listed = gantry_list()
    for job in listed:
        add(job)
    by_id = {job.id: job for job in listed}
    missing = {
        name: job_id
        for name, job_id in (recorded or {}).items()
        if name in names and job_id not in by_id
    }
    with ThreadPoolExecutor(LOOKUP_THREADS) as ex:
        for (name, _), looked_up in zip(
            missing.items(), ex.map(gantry_job, missing.values())
        ):
            if looked_up is not None:
                add(looked_up, name)
        listings = ex.map(
            lambda item: _complete_listing(item[0], set(item[1]), workspace),
            names_by_text.items(),
        )
        for listing in listings:
            for job in listing:
                # gantry's own view of its jobs is the fresher one.
                if job.id not in by_id:
                    add(job)
    return {name: list(jobs.values()) for name, jobs in found.items()}


def exists(jobs: Iterable[GantryJob]) -> bool:
    """Whether any of `jobs` succeeded or is running or queued (what
    --skip-if-in-beaker skips; failed and canceled ones leave the name free).
    """
    return any(gantry_status(job.status)[0] in (OK, RUNNING) for job in jobs)


def newest(jobs: Iterable[GantryJob]) -> GantryJob | None:
    """The most recently created of `jobs`, or None."""
    ordered = sorted(jobs, key=lambda job: job.created)
    return ordered[-1] if ordered else None
