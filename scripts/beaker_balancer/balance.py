"""Balance Beaker job priorities against the team's urgent slot allocation.

Beaker does not enforce our urgent slot allocation, so queueing urgent jobs can
silently put the team over it. This script reads the live state of a workspace
and adjusts the priority of opted-in jobs so that urgent slot usage stays within
the allocation, favouring the jobs we care about most.

A job opts in by setting the ``CM_PRIORITY`` environment variable to one of
``low``, ``normal``, ``high`` or ``urgent``. Jobs without it are never modified,
but their urgent slot usage still counts against the allocation. So do
interactive sessions and jobs at ``immediate`` priority, neither of which the
balancer will touch.

Decisions are made per *replica group*, not per job: Beaker runs a multi-node
job as one job per rank, and a group granted urgent on only some ranks holds
slots it cannot use.

Run ``python balance.py --dry-run`` to see what a pass would do.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from beaker import Beaker
from beaker import beaker_pb2 as pb2
from beaker._service_client import RpcMethod
from beaker.exceptions import BeakerError

LOGGER = logging.getLogger("beaker_balancer")

CM_PRIORITY_ENV = "CM_PRIORITY"

#: GPU slots the team may hold at urgent priority, per cluster. Clusters absent
#: from this mapping have no allocation and are left alone entirely.
DEFAULT_CLUSTER_LIMITS = {"ai2/jupiter": 72, "ai2/titan": 32}

#: Workspaces whose jobs spend the same allocation but are never modified. The
#: allocation is the team's, not one workspace's, so slots held here have to be
#: subtracted before any are handed out -- otherwise the balancer grants urgent
#: against slots that are already occupied. ``CM_PRIORITY`` is not read in
#: these: opting in is what makes a job the balancer's to move, and nothing
#: here is.
DEFAULT_OBSERVED_WORKSPACES = ("ai2/climate-titan",)

LOW = pb2.JOB_PRIORITY_LOW
NORMAL = pb2.JOB_PRIORITY_NORMAL
HIGH = pb2.JOB_PRIORITY_HIGH
URGENT = pb2.JOB_PRIORITY_URGENT
IMMEDIATE = pb2.JOB_PRIORITY_IMMEDIATE

#: ``immediate`` is deliberately absent: Beaker requires a human-supplied
#: reason for it, so it is not something the balancer may hand out.
PRIORITY_BY_NAME = {"low": LOW, "normal": NORMAL, "high": HIGH, "urgent": URGENT}
NAME_BY_PRIORITY = {value: name for name, value in PRIORITY_BY_NAME.items()}
NAME_BY_PRIORITY[IMMEDIATE] = "immediate"

#: Priorities that occupy a slot in the allocation. ``immediate`` outranks
#: ``urgent``, so a job holding one is just as much an occupant.
ALLOCATED_PRIORITIES = (URGENT, IMMEDIATE)

_CLUSTER_CONSTRAINT = pb2.JOB_PLACEMENT_CONSTRAINT_TYPE_CLUSTER

# Why a job may not be modified. Identity-compared, so a caller can tell an
# expected state apart from one worth warning about.
UNLABELLED = "does not set CM_PRIORITY"
OBSERVED = "is in an observed workspace, which is counted but never modified"
SESSION = "is an interactive session"
AT_IMMEDIATE = "is at immediate priority, which only a human sets"
NO_ALLOCATION = "targets a cluster with no allocation"
UNRESOLVED_CLUSTER = "landed on a node that cannot be resolved to a cluster"


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
    #: Whether Beaker has placed this job on a node. True from the moment it is
    #: scheduled, which is before it starts running -- an initializing job is
    #: already occupying its slots, and Beaker already refuses to raise its
    #: priority, so it must not be treated as queued.
    is_placed: bool
    #: When the job entered the queue. Reset when a job is preempted and
    #: requeued, so this measures the wait of the *current* attempt.
    queued_at: float
    #: When the job took its slots, or None if it is still queued.
    placed_at: float | None
    #: Size of the job's replica group, or 0 when it is a lone job.
    replica_group_size: int = 0
    #: Identifies the replicas that must start together. Empty for a lone job.
    replica_group_id: str = ""
    #: An interactive session has a person attached to it, so its priority is
    #: never ours to change.
    is_session: bool = False
    #: The workspace the job is in, for reporting.
    workspace: str = ""
    #: Whether the job is only being watched: its slots are counted against the
    #: allocation, but it is never modified and its CM_PRIORITY is not read.
    is_observed: bool = False

    @property
    def is_queued(self) -> bool:
        """Whether the job is still waiting, and so may have its priority raised."""
        return not self.is_placed

    @property
    def holds_allocation(self) -> bool:
        """Whether this job currently occupies a slot in the allocation."""
        return self.priority in ALLOCATED_PRIORITIES

    @property
    def group_key(self) -> str:
        """What this job is decided along with: its replica group, or itself."""
        if self.replica_group_size > 1 and self.replica_group_id:
            return self.replica_group_id
        return self.id

    def budget_clusters(self, limits: dict[str, int]) -> tuple[str, ...]:
        """Clusters whose allocation this job consumes while it is urgent.

        A placed job consumes only the cluster it landed on. A queued job could
        land on any cluster it is eligible for, and since nothing can preempt an
        urgent job it is a committed future occupant of whichever it picks, so
        it is counted against all of them.

        A job whose landing cluster cannot be narrowed -- placed but
        unresolvable, or queued with no cluster constraint at all -- is charged
        to every budgeted cluster. Charging it to none would let the balancer
        hand out slots that are already spoken for.
        """
        if self.is_placed:
            if self.assigned_cluster is None:
                return tuple(limits)
            candidates: tuple[str, ...] = (self.assigned_cluster,)
        elif not self.clusters:
            return tuple(limits)
        else:
            candidates = self.clusters
        return tuple(cluster for cluster in candidates if cluster in limits)

    def unmanaged_reason(self, limits: dict[str, int]) -> str | None:
        """Why the balancer may not change this job, or None if it may.

        Returns one of the module-level reason constants, so a caller can tell
        an expected state apart from one worth warning about.

        Replica group members are *not* excluded here. They are decided as a
        group rather than individually, which is what stops the walk granting
        urgent to some ranks and not others; see ``ReplicaGroup``.
        """
        if self.is_observed:
            return OBSERVED
        if self.cm_priority is None:
            return UNLABELLED
        if self.is_session:
            return SESSION
        if self.priority == IMMEDIATE:
            return AT_IMMEDIATE
        # A placed job is pinned in fact. It occupies one cluster and is
        # charged to that one alone, whatever it was submitted eligible for, so
        # what it was eligible for stops being a reason not to touch it. Only a
        # placed job whose node cannot be resolved is still charged to every
        # budget, and managing that would let the walk hand out slots on one
        # cluster while the accounting spent them on all of them.
        if self.is_placed and self.assigned_cluster is None:
            return UNRESOLVED_CLUSTER
        if not self.budget_clusters(limits):
            return NO_ALLOCATION
        return None

    def managed_clusters(self, limits: dict[str, int]) -> tuple[str, ...] | None:
        """The budgeted clusters this job is managed against, or None.

        For a placed job, a single cluster: the one it landed on. For a queued
        job, every budgeted cluster it could land on: granting it urgent
        commits the allocation on each, so the pass must check and charge all
        of them.
        """
        if self.unmanaged_reason(limits) is not None:
            return None
        return self.budget_clusters(limits)


@dataclass(frozen=True)
class Action:
    """A priority change the balancer intends to make."""

    job: JobView
    #: The budgeted clusters this change is accounted against.
    clusters: tuple[str, ...]
    from_priority: int
    to_priority: int
    reason: str

    @property
    def is_demotion(self) -> bool:
        return self.to_priority < self.from_priority

    @property
    def frees_slots(self) -> bool:
        """Whether applying this change returns slots to the allocation."""
        return self.from_priority == URGENT and self.to_priority != URGENT

    @property
    def takes_slots(self) -> bool:
        """Whether applying this change spends slots from the allocation."""
        return self.to_priority == URGENT and self.from_priority != URGENT

    def describe(self) -> str:
        return (
            f"{self.job.id} {self.job.name!r} ({self.job.author}, "
            f"{self.job.slots} slots): "
            f"{NAME_BY_PRIORITY.get(self.from_priority, self.from_priority)} -> "
            f"{NAME_BY_PRIORITY.get(self.to_priority, self.to_priority)} "
            f"[{self.reason}]"
        )


@dataclass(frozen=True)
class ReplicaGroup:
    """Jobs that must start together, granted or denied urgent as a unit.

    Beaker runs a multi-node job as one job per replica. Granting urgent to
    some ranks and not others is the worst available outcome: the group cannot
    start until every rank is placed, so it waits at the priority of its lowest
    rank while the granted ranks hold slots the allocation has already spent.

    A lone job is a group of one, so the walk has a single kind of candidate.
    """

    id: str
    jobs: tuple[JobView, ...]
    clusters: tuple[str, ...]
    cm_priority: int

    @property
    def slots(self) -> int:
        return sum(job.slots for job in self.jobs)

    @property
    def is_queued(self) -> bool:
        return all(job.is_queued for job in self.jobs)

    @property
    def is_placed(self) -> bool:
        return not self.is_queued

    @property
    def holds_urgent(self) -> bool:
        return all(job.priority == URGENT for job in self.jobs)

    @property
    def placed_at(self) -> float:
        """When the group first took slots."""
        return min((job.placed_at or 0.0) for job in self.jobs)

    @property
    def queued_at(self) -> float:
        """When the group last became wholly queued.

        The latest of its ranks, not the earliest: the group cannot start until
        the last one is placed. Since the clock resets when a rank is preempted
        and requeued, a group that has been bounced does not keep seniority it
        no longer has.
        """
        return max(job.queued_at for job in self.jobs)


def group_jobs(
    jobs: Sequence[JobView], limits: dict[str, int]
) -> tuple[list[ReplicaGroup], list[JobView]]:
    """Split jobs into manageable groups and jobs charged but left alone.

    A group is manageable only if *every* rank is individually manageable and
    they agree on their label and cluster. One rank the balancer may not touch
    would otherwise leave the group at mixed priorities, which is exactly what
    deciding as a group is meant to prevent.

    A group is also left alone unless every rank is present. ``fetch_jobs``
    reads only unfinished jobs in one workspace, so a group can come back
    partial; granting urgent to the ranks that happen to be visible would
    half-grant the real group.
    """
    by_group: dict[str, list[JobView]] = {}
    for job in jobs:
        by_group.setdefault(job.group_key, []).append(job)

    managed: list[ReplicaGroup] = []
    fixed: list[JobView] = []
    for group_id, members in by_group.items():
        cluster_tuples = {job.managed_clusters(limits) for job in members}
        labels = {job.cm_priority for job in members}
        # A single non-None value in each means every rank is manageable and
        # they agree; anything else leaves the group untouched.
        clusters = cluster_tuples.pop() if len(cluster_tuples) == 1 else None
        label = labels.pop() if len(labels) == 1 else None
        expected = max(job.replica_group_size for job in members)
        if clusters is None or label is None or len(members) < expected:
            fixed.extend(members)
            continue
        managed.append(
            ReplicaGroup(
                id=group_id,
                jobs=tuple(members),
                clusters=clusters,
                cm_priority=label,
            )
        )
    return managed, fixed


def _grant_order(groups: Sequence[ReplicaGroup]) -> list[ReplicaGroup]:
    """Order groups within one CM_PRIORITY level by who should keep urgent.

    Placed groups that already hold urgent come first, so a queued group is
    always given up before a running one at the same level. Among them the
    group that has held its slots longest sorts last, so it is the first to
    lose urgent. Queued groups follow, longest-waiting first.

    Both clocks are whole seconds and a launch wave shares one, so the group id
    breaks ties: without it the order would fall through to whatever sequence
    Beaker happened to list the jobs in, and the pass would not be reproducible.
    """
    placed = [group for group in groups if group.is_placed]
    queued = [group for group in groups if group.is_queued]
    # placed_at descending == shortest time holding slots first.
    placed.sort(key=lambda group: (-group.placed_at, group.id))
    # queued_at ascending == longest time waiting first.
    queued.sort(key=lambda group: (group.queued_at, group.id))
    return placed + queued


def _eligible_for_urgent(group: ReplicaGroup) -> bool:
    """Whether a group could hold urgent after this pass.

    Beaker refuses to raise the priority of a job it has already placed, so a
    group with a placed rank below urgent can never be brought wholly to
    urgent. Including one in the allocation would displace groups that can
    actually be promoted.
    """
    return group.holds_urgent or group.is_queued


def _reason(job: JobView, desired: int) -> str:
    """Why a job is being moved, in terms of the allocation.

    A job that neither takes nor gives up an urgent slot is only being put where
    its label says it rests -- a queued job submitted below its ``CM_PRIORITY``
    is raised to it. Describing that as releasing a slot would report a change
    to the allocation that did not happen.
    """
    if desired == URGENT:
        return "grant urgent slot"
    if job.priority == URGENT:
        return "release urgent slot"
    return "settle at resting priority"


def decide(jobs: Iterable[JobView], limits: dict[str, int]) -> list[Action]:
    """Compute the priority changes that bring urgent usage within allocation.

    The desired allocation is recomputed from scratch on every pass rather than
    adjusted incrementally: candidate groups are walked in CM_PRIORITY order
    and granted urgent while their cluster has slots left. Because the walk
    starts clean, a high-priority group is always considered before a
    lower-priority one and never has to wait for slots a lesser one took.
    """
    groups, fixed = group_jobs(list(jobs), limits)

    # Slots held at urgent or immediate by jobs the balancer will not touch.
    # These are a fixed charge against the allocation; only what is left over
    # is ours to hand out.
    remaining = dict(limits)
    for job in fixed:
        if not job.holds_allocation:
            continue
        for charged in job.budget_clusters(limits):
            remaining[charged] -= job.slots

    granted: set[str] = set()
    by_level: dict[int, list[ReplicaGroup]] = {}
    for group in groups:
        by_level.setdefault(group.cm_priority, []).append(group)

    for level in sorted(by_level, reverse=True):
        for group in _grant_order(by_level[level]):
            if not _eligible_for_urgent(group):
                continue
            # First fit: a group too large for what is left is skipped and
            # smaller ones behind it still get a chance. Nothing is reserved
            # for it, because a higher level that could not fit has already
            # taken everything it was entitled to. A multi-cluster group must
            # fit on every cluster it could land on: granting it urgent commits
            # the allocation on each, and nothing can preempt an urgent job.
            if all(group.slots <= remaining[c] for c in group.clusters):
                granted.add(group.id)
                for c in group.clusters:
                    remaining[c] -= group.slots

    actions = []
    for group in groups:
        desired = URGENT if group.id in granted else resting_priority(group.cm_priority)
        for job in group.jobs:
            if desired == job.priority:
                continue
            if job.is_placed and desired > job.priority:
                # Beaker rejects this; skip rather than log a failure every
                # pass. A group is only granted urgent when every rank can get
                # there, so this never splits a granted group.
                LOGGER.debug(
                    "%s is placed at %s and cannot be raised to %s",
                    job.id,
                    NAME_BY_PRIORITY.get(job.priority),
                    NAME_BY_PRIORITY.get(desired),
                )
                continue
            reason = _reason(job, desired)
            actions.append(Action(job, group.clusters, job.priority, desired, reason))

    # Demotions first, so that any prefix of a partially applied pass is still
    # within allocation. A promotion applied before the demotion paying for it
    # would leave us over if that demotion then failed.
    actions.sort(key=lambda action: not action.is_demotion)
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


def _cluster_of_node(
    client: Beaker, node_id: str, cache: dict[str, str | None]
) -> str | None:
    """Resolve the node's cluster to an org-qualified name, e.g. ``ai2/jupiter``.

    ``Cluster.name`` is bare, but placement constraints and our allocation are
    written org-qualified, so the two have to be reconciled or placed jobs match
    no budget and their slots go uncounted.

    Failures are cached too, so one unresolvable node is not retried for every
    job sitting on it. ``fetch_jobs`` drops them between passes.
    """
    if node_id not in cache:
        try:
            node = client.node.get(node_id)
            cluster = client.cluster.get(node.cluster_id)
            cache[node_id] = f"{cluster.organization_name}/{cluster.name}"
        except BeakerError as err:
            LOGGER.warning("could not resolve cluster for node %s: %s", node_id, err)
            cache[node_id] = None
    return cache[node_id]


def fetch_jobs(
    client: Beaker,
    workspace_name: str,
    node_cache: dict[str, str | None] | None = None,
    observed: Sequence[str] = (),
) -> list[JobView]:
    """Read every unfinished job in the managed and observed workspaces.

    Jobs from an observed workspace are marked ``is_observed``: their slots are
    counted against the allocation but they are never modified, and their
    ``CM_PRIORITY`` is not read -- setting it in a workspace the balancer does
    not manage would otherwise look like it did something.
    """
    workspace = client.workspace.get(workspace_name)
    # One listing covers the whole org, so watching another workspace costs a
    # name lookup rather than a second pass over the jobs.
    names = {workspace.id: workspace_name}
    for name in observed or ():
        names.setdefault(client.workspace.get(name).id, name)
    node_clusters: dict[str, str | None] = {} if node_cache is None else node_cache
    # Failures are cached within a pass so one bad node is not retried for every
    # job on it, but they are dropped between passes. A job whose node cannot be
    # resolved is charged to every budget, so a single transient Beaker error
    # would otherwise hold it there for the life of an --interval process --
    # under-granting the allocation, and silently, since the warning is logged
    # only the first time.
    for stale in [node for node, cluster in node_clusters.items() if cluster is None]:
        del node_clusters[stale]
    views = []
    for job in client.job.list(finalized=False):
        if job.workspace_id not in names:
            continue
        is_observed = job.workspace_id != workspace.id
        cm_priority = None
        if not is_observed:
            env = {
                var.name: var.literal
                for var in job.container_spec.environment_variables
            }
            if CM_PRIORITY_ENV in env:
                cm_priority = parse_cm_priority(env[CM_PRIORITY_ENV], job.id)

        clusters = tuple(
            value
            for constraint in job.system_details.placement_constraints
            if constraint.type == _CLUSTER_CONSTRAINT
            for value in constraint.values
        )
        node_id = job.assignment_details.node_id
        is_placed = bool(node_id)
        assigned = _cluster_of_node(client, node_id, node_clusters) if node_id else None
        # A scheduled job holds its slots before it starts running, so fall back
        # to the scheduling time rather than treating it as never placed.
        placed_at = job.status.started.seconds or job.status.scheduled.seconds or None

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
                is_placed=is_placed,
                queued_at=float(job.status.created.seconds),
                placed_at=float(placed_at) if is_placed and placed_at else None,
                replica_group_size=job.system_details.replica_group_details.size,
                replica_group_id=job.system_details.replica_group_details.id,
                # Only an interactive session carries an environment.
                is_session=bool(job.environment_id),
                workspace=names[job.workspace_id],
                is_observed=is_observed,
            )
        )
    return views


def set_priority(client: Beaker, job_id: str, priority: int) -> None:
    """Change one job's priority. The only call the balancer makes that mutates.

    ``beaker-py`` exposes no wrapper for ``UpdateJobSourcePriority``, so the rpc
    is issued directly through the private ``RpcMethod``. A beaker-py upgrade
    can therefore break this import, and CI will not notice: it does not install
    beaker-py, so both test modules skip. Run the tests locally after upgrading.
    """
    service = client.job
    service.rpc_request(
        RpcMethod(service.service.UpdateJobSourcePriority),
        pb2.UpdateJobSourcePriorityRequest(
            job_id=job_id, priority=priority, reason="beaker_balancer"
        ),
    )


def urgent_usage(jobs: Sequence[JobView], limits: dict[str, int]) -> dict[str, int]:
    """Slots occupied per cluster, counting immediate alongside urgent."""
    used = dict.fromkeys(limits, 0)
    for job in jobs:
        if not job.holds_allocation:
            continue
        for cluster in job.budget_clusters(limits):
            used[cluster] += job.slots
    return used


def report_usage(jobs: Sequence[JobView], limits: dict[str, int], label: str) -> None:
    used = urgent_usage(jobs, limits)
    summary = ", ".join(
        f"{cluster} {used[cluster]}/{limit}" for cluster, limit in limits.items()
    )
    LOGGER.info("urgent slots %s: %s", label, summary)


def report_unreclaimable(jobs: Sequence[JobView], limits: dict[str, int]) -> None:
    """Break out allocated slots the balancer can never reclaim.

    Sessions, immediate-priority jobs and jobs in an observed workspace count
    against the allocation but are never demoted, so a pass can be unable to get
    within the allocation no matter what it does. Reporting them makes that a
    visible fact rather than a balancer that appears not to be working.
    """
    sessions = dict.fromkeys(limits, 0)
    immediate = dict.fromkeys(limits, 0)
    elsewhere: dict[str, Counter[str]] = {cluster: Counter() for cluster in limits}
    for job in jobs:
        if not job.holds_allocation:
            continue
        for cluster in job.budget_clusters(limits):
            # Checked first: a session or immediate job in an observed
            # workspace is unreclaimable because of where it is, and reporting
            # it under its own workspace is what makes the number add up.
            if job.is_observed:
                elsewhere[cluster][job.workspace] += job.slots
            elif job.is_session:
                sessions[cluster] += job.slots
            elif job.priority == IMMEDIATE:
                immediate[cluster] += job.slots
    for cluster in limits:
        notes = []
        if sessions[cluster]:
            notes.append(f"{sessions[cluster]} in interactive sessions")
        if immediate[cluster]:
            notes.append(f"{immediate[cluster]} at immediate priority")
        for workspace, slots in sorted(elsewhere[cluster].items()):
            notes.append(f"{slots} in {workspace}")
        if notes:
            LOGGER.info(
                "%s: %s — counted against the allocation, never reclaimable",
                cluster,
                "; ".join(notes),
            )


def report_unmanageable(jobs: Sequence[JobView], limits: dict[str, int]) -> None:
    """Summarise opted-in jobs the balancer cannot act on.

    Reported as one aggregate line rather than one per job: most jobs in the
    workspace target several clusters, so per-job warnings would drown the log
    during adoption.
    """
    reasons: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for job in jobs:
        why = job.unmanaged_reason(limits)
        # An observed job is reported as held slots, not as a job that failed
        # to be managed: it was never a candidate, and there can be many.
        if why is None or why is UNLABELLED or why is OBSERVED:
            continue
        if why is NO_ALLOCATION:
            reason = f"targets {job.clusters[0]}, which has no allocation"
        else:
            reason = why
        reasons[reason] += 1
        examples.setdefault(reason, job.id)

    # A group left alone as a whole is not visible per job, since each rank may
    # be individually fine.
    managed_ids = {
        job.id for group in group_jobs(jobs, limits)[0] for job in group.jobs
    }
    skipped_groups = {
        job.group_key
        for job in jobs
        if job.replica_group_size > 1
        and job.id not in managed_ids
        and job.unmanaged_reason(limits) is None
    }
    if skipped_groups:
        LOGGER.warning(
            "%d replica group(s) left alone because their ranks are not "
            "uniformly manageable (e.g. %s)",
            len(skipped_groups),
            sorted(skipped_groups)[0],
        )
    for reason, count in sorted(reasons.items()):
        LOGGER.warning(
            "%d job(s) set %s but cannot be managed: %s (e.g. %s)",
            count,
            CM_PRIORITY_ENV,
            reason,
            examples[reason],
        )


def run_pass(
    client: Beaker,
    workspace: str,
    limits: dict[str, int],
    dry_run: bool,
    node_cache: dict[str, str | None] | None = None,
    observed: Sequence[str] = (),
) -> int:
    """Run one balancing pass. Returns the number of changes applied."""
    jobs = fetch_jobs(client, workspace, node_cache, observed)
    watched = sum(1 for job in jobs if job.is_observed)
    LOGGER.info(
        "read %d unfinished jobs in %s%s",
        len(jobs) - watched,
        workspace,
        f", and {watched} watched in {', '.join(observed)}" if watched else "",
    )
    report_usage(jobs, limits, "before")
    report_unreclaimable(jobs, limits)
    report_unmanageable(jobs, limits)

    actions = decide(jobs, limits)
    if not actions:
        LOGGER.info("no changes needed")
        return 0

    applied = 0
    changed: dict[str, int] = {}
    # Slots a refused demotion failed to free, per cluster. Ordering demotions
    # first makes any *prefix* of a pass safe, but a failure does not stop the
    # pass -- it skips one action and carries on -- so what lands is an
    # arbitrary subset, and the subset that matters is exactly the documented
    # fail-soft case. Granting urgent after the demotion paying for it was
    # refused ends the pass over allocation. Such a grant is deferred to the
    # next pass, which recomputes from the state that really exists. The
    # deficit is per cluster, so a stuck job on one does not stall the other.
    deficit: Counter[str] = Counter()
    # What each replica group is waiting to be granted. A group is granted all
    # or nothing, so it has to be deferred all or nothing too.
    wanted: Counter[str] = Counter()
    for action in actions:
        if action.takes_slots:
            wanted[action.job.group_key] += action.job.slots
    # Groups whose grant this pass has given up on.
    abandoned: set[str] = set()

    for action in actions:
        if dry_run:
            LOGGER.info("would change %s", action.describe())
            continue
        if action.takes_slots:
            if action.job.group_key in abandoned:
                continue
            deficient = [c for c in action.clusters if deficit[c] > 0]
            if deficient:
                for c in action.clusters:
                    deficit[c] -= wanted[action.job.group_key]
                abandoned.add(action.job.group_key)
                LOGGER.warning(
                    "not granting urgent to %s: a demotion on %s was refused, "
                    "so the slots paying for it are still held",
                    action.job.group_key,
                    ", ".join(deficient),
                )
                continue
        try:
            set_priority(client, action.job.id, action.to_priority)
        except BeakerError as err:
            # Most likely a job owned by someone whose jobs we cannot modify.
            # One unreachable job should not stop the rest of the pass.
            if action.frees_slots:
                for c in action.clusters:
                    deficit[c] += action.job.slots
            if action.takes_slots:
                # Stop before splitting the group any further. Permission is
                # per owner and every rank shares one, so in practice this
                # fires on the first rank and spends nothing. A rank already
                # granted is left for the next pass to finish or undo, rather
                # than adding a rollback path that can fail in its turn.
                abandoned.add(action.job.group_key)
            LOGGER.warning(
                "could not change %s (owner %s): %s",
                action.job.id,
                action.job.author,
                err,
            )
            continue
        applied += 1
        changed[action.job.id] = action.to_priority
        LOGGER.info("changed %s", action.describe())

    if not dry_run:
        settled = [
            replace(job, priority=changed[job.id]) if job.id in changed else job
            for job in jobs
        ]
        report_usage(settled, limits, "after")
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
        help="override one cluster's urgent GPU slot allocation; repeatable. "
        "Other clusters keep their defaults: "
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
        "--observe",
        action="append",
        metavar="WORKSPACE",
        help="also count this workspace's urgent slots against the allocation, "
        "without ever modifying its jobs; repeatable. Replaces the default: "
        + ", ".join(DEFAULT_OBSERVED_WORKSPACES),
    )
    parser.add_argument(
        "--no-observe",
        action="store_true",
        help="count only the managed workspace's slots",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="include debug logging"
    )
    return parser


def parse_limits(values: list[str] | None) -> dict[str, int]:
    """Apply CLUSTER=SLOTS overrides on top of the default allocation.

    Overrides merge rather than replace: naming one cluster must not silently
    drop the allocation of another and leave its jobs unmanaged.
    """
    limits = dict(DEFAULT_CLUSTER_LIMITS)
    for value in values or []:
        cluster, sep, slots = value.partition("=")
        if not sep or not cluster or not slots.isdigit():
            raise ValueError(f"expected CLUSTER=SLOTS, got {value!r}")
        limits[cluster] = int(slots)
    return limits


def validate_limits(client: Beaker, limits: dict[str, int]) -> None:
    """Check every budgeted cluster name resolves.

    A typo here manages nothing at all and looks exactly like a quiet cluster,
    so fail loudly at startup instead.
    """
    for cluster in limits:
        try:
            client.cluster.get(cluster)
        except BeakerError as err:
            raise SystemExit(f"unknown cluster {cluster!r} in allocation: {err}")


def validate_workspaces(client: Beaker, workspaces: Sequence[str]) -> None:
    """Check every workspace name resolves.

    A typo in an observed workspace counts nothing and looks exactly like a
    workspace holding no urgent slots, so the balancer would quietly hand out
    the allocation twice.
    """
    for workspace in workspaces:
        try:
            client.workspace.get(workspace)
        except BeakerError as err:
            raise SystemExit(f"unknown workspace {workspace!r}: {err}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        limits = parse_limits(args.limit)
    except ValueError as err:
        parser.error(str(err))
    LOGGER.info(
        "allocation: %s", ", ".join(f"{k}={v} slots" for k, v in limits.items())
    )
    observed: tuple[str, ...] = ()
    if not args.no_observe:
        observed = tuple(args.observe or DEFAULT_OBSERVED_WORKSPACES)
    if observed:
        LOGGER.info("also counting, but never modifying: %s", ", ".join(observed))

    client = Beaker.from_env()
    validate_limits(client, limits)
    validate_workspaces(client, [args.workspace, *observed])
    node_cache: dict[str, str | None] = {}
    while True:
        try:
            run_pass(client, args.workspace, limits, args.dry_run, node_cache, observed)
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
