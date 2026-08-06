"""Balance Beaker job priorities against the team's urgent slot allocation.

Beaker does not enforce our per-cluster urgent allocation, so queueing urgent
jobs can silently put the team over it. This script reads the live state of a
workspace and adjusts the priority of opted-in jobs so that urgent slot usage
stays within the allocation, favouring the jobs we care about most.

A job opts in by setting the ``CM_PRIORITY`` environment variable to one of
``low``, ``normal``, ``high`` or ``urgent``. Jobs without it are never
modified, but their urgent slot usage still counts against the allocation.

Run ``python balance.py --dry-run`` to see what a pass would do.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from beaker import Beaker
from beaker import beaker_pb2 as pb2
from beaker._service_client import RpcMethod
from beaker.exceptions import BeakerError

LOGGER = logging.getLogger("beaker_balancer")

CM_PRIORITY_ENV = "CM_PRIORITY"

#: GPU slots the team may hold at urgent priority, per cluster. Clusters absent
#: from this mapping have no allocation and are left alone entirely.
DEFAULT_CLUSTER_LIMITS = {"ai2/jupiter": 72, "ai2/titan": 32}

LOW = pb2.JOB_PRIORITY_LOW
NORMAL = pb2.JOB_PRIORITY_NORMAL
HIGH = pb2.JOB_PRIORITY_HIGH
URGENT = pb2.JOB_PRIORITY_URGENT

PRIORITY_BY_NAME = {"low": LOW, "normal": NORMAL, "high": HIGH, "urgent": URGENT}
NAME_BY_PRIORITY = {value: name for name, value in PRIORITY_BY_NAME.items()}

# A job's status is "queued" until Beaker places it on a node. Only queued jobs
# can have their priority raised; Beaker rejects increases on running jobs with
# "cannot increase priority for running job".
_CLUSTER_CONSTRAINT = pb2.JOB_PLACEMENT_CONSTRAINT_TYPE_CLUSTER


def resting_priority(cm_priority: int) -> int:
    """Priority a job falls back to when it is not granted an urgent slot.

    Jobs labelled ``urgent`` rest at ``high``: resting at urgent would defeat
    the purpose of the allocation.
    """
    return HIGH if cm_priority == URGENT else cm_priority


@dataclass(frozen=True)
class JobView:
    """The parts of a Beaker job the balancer reasons about.

    Kept free of protobuf types so the decision logic can be unit tested with
    plain constructed values.
    """

    id: str
    name: str
    author: str
    priority: int
    slots: int
    clusters: tuple[str, ...]
    assigned_cluster: str | None
    cm_priority: int | None
    #: When the job entered the queue. Reset when a job is preempted and
    #: requeued, so this measures the wait of the *current* attempt.
    queued_at: float
    #: When the job landed on a GPU, or None if it is still queued.
    started_at: float | None

    @property
    def is_queued(self) -> bool:
        return self.started_at is None

    def budget_clusters(self, limits: dict[str, int]) -> tuple[str, ...]:
        """Clusters whose allocation this job consumes while it is urgent.

        An assigned job consumes only the cluster it actually landed on. A
        queued job could land on any cluster it is eligible for, and since
        nothing can preempt an urgent job it is a committed future occupant of
        whichever it picks, so it is counted against all of them.
        """
        if self.assigned_cluster is not None:
            candidates: tuple[str, ...] = (self.assigned_cluster,)
        else:
            candidates = self.clusters
        return tuple(cluster for cluster in candidates if cluster in limits)

    def managed_cluster(self, limits: dict[str, int]) -> str | None:
        """The single budgeted cluster this job may be modified against.

        The balancer only ever changes jobs pinned to exactly one budgeted
        cluster. A job eligible for several could land on any of them, and
        Beaker offers no way to pin it, so its effect on a given allocation is
        not knowable in advance.
        """
        if self.cm_priority is None or len(self.clusters) != 1:
            return None
        cluster = self.clusters[0]
        return cluster if cluster in limits else None


@dataclass(frozen=True)
class Action:
    """A priority change the balancer intends to make."""

    job: JobView
    from_priority: int
    to_priority: int
    reason: str

    def describe(self) -> str:
        return (
            f"{self.job.id} {self.job.name!r} ({self.job.author}, "
            f"{self.job.slots} slots): "
            f"{NAME_BY_PRIORITY.get(self.from_priority, self.from_priority)} -> "
            f"{NAME_BY_PRIORITY.get(self.to_priority, self.to_priority)} "
            f"[{self.reason}]"
        )


def _grant_order(jobs: Sequence[JobView]) -> list[JobView]:
    """Order jobs within one CM_PRIORITY level by who should keep urgent.

    Running jobs that already hold urgent come first, so a queued job is always
    given up before a running one at the same level. Among them the job that
    has held a GPU longest sorts last, so it is the first to lose urgent.
    Queued jobs follow, longest-waiting first.
    """
    running = [job for job in jobs if not job.is_queued]
    queued = [job for job in jobs if job.is_queued]
    # started_at descending == shortest time on GPU first.
    running.sort(key=lambda job: job.started_at or 0.0, reverse=True)
    # queued_at ascending == longest time waiting first.
    queued.sort(key=lambda job: job.queued_at)
    return running + queued


def _eligible_for_urgent(job: JobView) -> bool:
    """Whether a job could hold urgent after this pass.

    Beaker refuses to raise the priority of a running job, so a running job
    that is not already urgent can never be promoted. Including such a job in
    the allocation would displace others to no benefit.
    """
    return job.priority == URGENT or job.is_queued


def decide(jobs: Iterable[JobView], limits: dict[str, int]) -> list[Action]:
    """Compute the priority changes that bring urgent usage within allocation.

    The desired allocation is recomputed from scratch on every pass rather than
    adjusted incrementally: candidates are walked in CM_PRIORITY order and
    granted urgent while their cluster has slots left. Because the walk starts
    clean, a high-priority job is always considered before a lower-priority one
    and never has to wait for slots a lesser job already took.
    """
    all_jobs = list(jobs)
    # Resolve each managed job's cluster and label once: a job is managed
    # exactly when both are known.
    managed: list[tuple[JobView, str, int]] = []
    for job in all_jobs:
        cluster = job.managed_cluster(limits)
        if cluster is not None and job.cm_priority is not None:
            managed.append((job, cluster, job.cm_priority))
    managed_ids = {job.id for job, _, _ in managed}

    # Slots held at urgent by jobs the balancer will not touch. These are a
    # fixed charge against the allocation; only what is left over is ours to
    # hand out.
    remaining = dict(limits)
    for job in all_jobs:
        if job.id in managed_ids or job.priority != URGENT:
            continue
        for charged in job.budget_clusters(limits):
            remaining[charged] -= job.slots

    granted: set[str] = set()
    by_level: dict[int, list[tuple[JobView, str]]] = {}
    for job, cluster, level in managed:
        by_level.setdefault(level, []).append((job, cluster))

    for level in sorted(by_level, reverse=True):
        candidates = {job.id: cluster for job, cluster in by_level[level]}
        for job in _grant_order([job for job, _ in by_level[level]]):
            if not _eligible_for_urgent(job):
                continue
            cluster = candidates[job.id]
            # First fit: a job too large for what is left is skipped and
            # smaller ones behind it still get a chance. Nothing is reserved
            # for it, because a higher level that could not fit has already
            # taken everything it was entitled to.
            if job.slots <= remaining[cluster]:
                granted.add(job.id)
                remaining[cluster] -= job.slots

    actions = []
    for job, _, level in managed:
        desired = URGENT if job.id in granted else resting_priority(level)
        if desired == job.priority:
            continue
        if not job.is_queued and desired > job.priority:
            # Beaker rejects this; skip rather than log a failure every pass.
            LOGGER.debug(
                "%s is running at %s and cannot be raised to %s",
                job.id,
                NAME_BY_PRIORITY.get(job.priority),
                NAME_BY_PRIORITY.get(desired),
            )
            continue
        reason = "grant urgent slot" if desired == URGENT else "release urgent slot"
        actions.append(Action(job, job.priority, desired, reason))
    return actions


def parse_cm_priority(value: str, job_id: str) -> int | None:
    """Parse a CM_PRIORITY value, returning None if it is not usable."""
    priority = PRIORITY_BY_NAME.get(value.strip().lower())
    if priority is None:
        LOGGER.warning(
            "%s sets %s=%r, which is not one of %s; ignoring the job",
            job_id,
            CM_PRIORITY_ENV,
            value,
            ", ".join(PRIORITY_BY_NAME),
        )
    return priority


def _cluster_of_node(client: Beaker, node_id: str, cache: dict[str, str]) -> str | None:
    """Resolve the node's cluster to an org-qualified name, e.g. ``ai2/jupiter``.

    ``Cluster.name`` is bare, but placement constraints and our allocation are
    written org-qualified, so the two have to be reconciled or assigned jobs
    match no budget and their slots go uncounted.
    """
    if node_id not in cache:
        try:
            node = client.node.get(node_id)
            cluster = client.cluster.get(node.cluster_id)
            cache[node_id] = f"{cluster.organization_name}/{cluster.name}"
        except BeakerError as err:
            LOGGER.warning("could not resolve cluster for node %s: %s", node_id, err)
            return None
    return cache[node_id]


def fetch_jobs(client: Beaker, workspace_name: str) -> list[JobView]:
    """Read every unfinished job in a workspace into JobView form."""
    workspace = client.workspace.get(workspace_name)
    node_clusters: dict[str, str] = {}
    views = []
    for job in client.job.list(finalized=False):
        if job.workspace_id != workspace.id:
            continue
        env = {
            var.name: var.literal for var in job.container_spec.environment_variables
        }
        cm_priority = None
        if CM_PRIORITY_ENV in env:
            cm_priority = parse_cm_priority(env[CM_PRIORITY_ENV], job.id)

        clusters = tuple(
            value
            for constraint in job.system_details.placement_constraints
            if constraint.type == _CLUSTER_CONSTRAINT
            for value in constraint.values
        )
        node_id = job.assignment_details.node_id
        assigned = _cluster_of_node(client, node_id, node_clusters) if node_id else None
        started = job.status.started.seconds or None

        views.append(
            JobView(
                id=job.id,
                name=job.name,
                author=job.author_reference,
                priority=job.system_details.priority,
                # Beaker counts a CPU-only job as one slot.
                slots=max(1, job.container_spec.resource_request.gpu_count),
                clusters=clusters,
                assigned_cluster=assigned,
                cm_priority=cm_priority,
                queued_at=float(job.status.created.seconds),
                started_at=float(started) if started else None,
            )
        )
    return views


def set_priority(client: Beaker, job_id: str, priority: int) -> None:
    service = client.job
    service.rpc_request(
        RpcMethod(service.service.UpdateJobSourcePriority),
        pb2.UpdateJobSourcePriorityRequest(
            job_id=job_id, priority=priority, reason="beaker_balancer"
        ),
    )


def report_usage(jobs: Sequence[JobView], limits: dict[str, int], label: str) -> None:
    used = dict.fromkeys(limits, 0)
    for job in jobs:
        if job.priority != URGENT:
            continue
        for cluster in job.budget_clusters(limits):
            used[cluster] += job.slots
    summary = ", ".join(
        f"{cluster} {used[cluster]}/{limit}" for cluster, limit in limits.items()
    )
    LOGGER.info("urgent slots %s: %s", label, summary)


def warn_unmanageable(jobs: Sequence[JobView], limits: dict[str, int]) -> None:
    """Flag opted-in jobs the balancer cannot act on, so the label isn't silent."""
    for job in jobs:
        if job.cm_priority is None or job.managed_cluster(limits) is not None:
            continue
        if len(job.clusters) != 1:
            LOGGER.warning(
                "%s %r sets %s but targets %d clusters (%s); the balancer only "
                "manages jobs pinned to one cluster, so it is left alone",
                job.id,
                job.name,
                CM_PRIORITY_ENV,
                len(job.clusters),
                ", ".join(job.clusters) or "none",
            )
        else:
            LOGGER.info(
                "%s %r sets %s but targets %s, which has no allocation; "
                "leaving it alone",
                job.id,
                job.name,
                CM_PRIORITY_ENV,
                job.clusters[0],
            )


def run_pass(
    client: Beaker, workspace: str, limits: dict[str, int], dry_run: bool
) -> int:
    """Run one balancing pass. Returns the number of changes applied."""
    jobs = fetch_jobs(client, workspace)
    LOGGER.info("read %d unfinished jobs in %s", len(jobs), workspace)
    report_usage(jobs, limits, "before")
    warn_unmanageable(jobs, limits)

    actions = decide(jobs, limits)
    if not actions:
        LOGGER.info("no changes needed")
        return 0

    applied = 0
    for action in actions:
        if dry_run:
            LOGGER.info("would change %s", action.describe())
            continue
        try:
            set_priority(client, action.job.id, action.to_priority)
        except BeakerError as err:
            # Most likely a job owned by someone whose jobs we cannot modify.
            # One unreachable job should not stop the rest of the pass.
            LOGGER.warning(
                "could not change %s (owner %s): %s",
                action.job.id,
                action.job.author,
                err,
            )
            continue
        applied += 1
        LOGGER.info("changed %s", action.describe())
    return applied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", default="ai2/ace", help="workspace to balance (default: ai2/ace)"
    )
    parser.add_argument(
        "--limit",
        action="append",
        metavar="CLUSTER=SLOTS",
        help="urgent GPU slot allocation for a cluster; repeatable. Defaults to "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_CLUSTER_LIMITS.items()),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the changes a pass would make without making them",
    )
    parser.add_argument(
        "--interval",
        type=float,
        metavar="SECONDS",
        help="keep running, pausing this long between passes (default: one pass)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="include debug logging"
    )
    return parser


def parse_limits(values: list[str] | None) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_CLUSTER_LIMITS)
    limits = {}
    for value in values:
        cluster, _, slots = value.partition("=")
        if not slots.isdigit():
            raise ValueError(f"expected CLUSTER=SLOTS, got {value!r}")
        limits[cluster] = int(slots)
    return limits


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    limits = parse_limits(args.limit)
    LOGGER.info(
        "allocation: %s", ", ".join(f"{k}={v} slots" for k, v in limits.items())
    )

    client = Beaker.from_env()
    while True:
        try:
            run_pass(client, args.workspace, limits, args.dry_run)
        except BeakerError:
            if args.interval is None:
                raise
            # A transient Beaker outage should not kill a long-running loop.
            LOGGER.exception("pass failed; retrying after %s seconds", args.interval)
        if args.interval is None:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
