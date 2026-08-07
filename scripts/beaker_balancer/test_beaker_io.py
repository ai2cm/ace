"""Integration tests for the balancer's Beaker-facing layer.

These build real protobuf messages of the shapes Beaker returns and drive
``fetch_jobs``, ``set_priority`` and ``run_pass`` against a fake client, so the
parsing and mutation paths are covered without touching a live workspace.
"""

import json
import logging
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("beaker", reason="beaker-py is not a core fme dependency")

import balance  # noqa: E402
from balance import (  # noqa: E402
    HIGH,
    LOW,
    NORMAL,
    URGENT,
    _cluster_of_node,
    fetch_jobs,
    parse_cm_priority,
    parse_limits,
    run_pass,
    set_priority,
    urgent_usage,
    validate_limits,
)
from beaker import beaker_pb2 as pb2  # noqa: E402
from beaker.exceptions import BeakerError  # noqa: E402

WORKSPACE_ID = "ws-ace"
LIMITS = {"ai2/jupiter": 72, "ai2/titan": 32}


@pytest.fixture(autouse=True)
def state_in_tmp_path(tmp_path, monkeypatch):
    """Keep every pass's remembered resting priorities inside the test.

    ``run_pass`` defaults to a real path under the user's home directory, so
    without this the suite would read and overwrite the memory of a balancer
    actually running on this machine -- and leak state between tests.
    """
    path = tmp_path / "resting.json"
    monkeypatch.setattr(balance, "DEFAULT_STATE_PATH", path)
    return path


def make_job(
    job_id="job-1",
    name="train",
    author="jeremym",
    priority=pb2.JOB_PRIORITY_HIGH,
    gpu_count=8,
    clusters=("ai2/jupiter",),
    hostnames=(),
    env=None,
    node_id="",
    created=1000,
    scheduled=0,
    started=0,
    replica_size=0,
    replica_group_id="",
    environment_id="",
    workspace_id=WORKSPACE_ID,
    retry_ancestor_id="",
):
    """Build a pb2.Job mirroring the shape Beaker returns for a live job."""
    job = pb2.Job(
        id=job_id,
        name=name,
        author_reference=author,
        workspace_id=workspace_id,
        task_id="task-1",
        workload_id="workload-1",
        environment_id=environment_id,
        retry_ancestor_id=retry_ancestor_id,
    )
    job.container_spec.resource_request.gpu_count = gpu_count
    for key, value in (env or {}).items():
        job.container_spec.environment_variables.add(name=key, literal=value)

    job.system_details.priority = priority
    if clusters:
        job.system_details.placement_constraints.add(
            type=pb2.JOB_PLACEMENT_CONSTRAINT_TYPE_CLUSTER, values=list(clusters)
        )
    if hostnames:
        job.system_details.placement_constraints.add(
            type=pb2.JOB_PLACEMENT_CONSTRAINT_TYPE_HOSTNAME, values=list(hostnames)
        )
    if replica_size:
        job.system_details.replica_group_details.size = replica_size
    if replica_group_id:
        job.system_details.replica_group_details.id = replica_group_id

    job.status.created.seconds = created
    if scheduled:
        job.status.scheduled.seconds = scheduled
    if started:
        job.status.started.seconds = started
        job.status.status = pb2.STATUS_RUNNING
    elif scheduled:
        job.status.status = pb2.STATUS_INITIALIZING
    else:
        job.status.status = pb2.STATUS_QUEUED
    if node_id:
        job.assignment_details.node_id = node_id
    return job


class FakeClient:
    """Stands in for beaker.Beaker over the surface the balancer uses."""

    def __init__(self, jobs, node_clusters=None, fail_on=(), unknown_clusters=()):
        self._jobs = list(jobs)
        # node id -> (bare cluster name, org name)
        self._node_clusters = node_clusters or {"node-jup": ("jupiter", "ai2")}
        self._fail_on = set(fail_on)
        self._unknown_clusters = set(unknown_clusters)
        self.priority_calls: list[tuple[str, int]] = []
        self.list_kwargs: list[dict] = []
        self.node_lookups: list[str] = []

        outer = self

        class _JobService:
            service = SimpleNamespace(UpdateJobSourcePriority=object())

            def list(self, **kwargs):
                outer.list_kwargs.append(kwargs)
                return iter(outer._jobs)

            def rpc_request(self, method, request, **kwargs):
                if request.job_id in outer._fail_on:
                    raise BeakerError(f"cannot modify {request.job_id}")
                outer.priority_calls.append((request.job_id, request.priority))
                assert request.reason, "a reason should be recorded on every change"
                return pb2.UpdateJobSourcePriorityResponse()

        self.job = _JobService()
        self.workspace = SimpleNamespace(
            get=lambda name: SimpleNamespace(id=WORKSPACE_ID, name=name)
        )
        self.node = SimpleNamespace(get=self._get_node)
        self.cluster = SimpleNamespace(get=self._get_cluster)

    def _get_node(self, node_id):
        self.node_lookups.append(node_id)
        if node_id not in self._node_clusters:
            raise BeakerError(f"no such node {node_id}")
        return SimpleNamespace(cluster_id=node_id)

    def _get_cluster(self, cluster_id):
        if cluster_id in self._unknown_clusters:
            raise BeakerError(f"no such cluster {cluster_id}")
        if cluster_id in self._node_clusters:
            name, org = self._node_clusters[cluster_id]
            return SimpleNamespace(name=name, organization_name=org)
        org, _, name = cluster_id.partition("/")
        return SimpleNamespace(name=name, organization_name=org)


# --- fetch_jobs -------------------------------------------------------------


def test_fetch_reads_a_queued_job():
    client = FakeClient([make_job(env={"CM_PRIORITY": "high"}, created=1234)])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.id == "job-1"
    assert view.author == "jeremym"
    assert view.priority == HIGH
    assert view.cm_priority == HIGH
    assert view.slots == 8
    assert view.clusters == ("ai2/jupiter",)
    assert view.assigned_cluster is None
    assert view.is_queued
    assert view.queued_at == 1234.0
    assert view.placed_at is None


def test_fetch_only_reads_unfinished_jobs():
    # Dropping this filter would start charging finished jobs' slots against
    # the allocation.
    client = FakeClient([make_job()])
    fetch_jobs(client, "ai2/ace")
    assert client.list_kwargs == [{"finalized": False}]


def test_fetch_qualifies_the_assigned_cluster_with_its_org():
    # Regression: Cluster.name is bare, so an unqualified value would match no
    # budget and the job's urgent slots would go uncounted.
    client = FakeClient(
        [make_job(node_id="node-jup", scheduled=4000, started=5000)],
        node_clusters={"node-jup": ("jupiter", "ai2")},
    )
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.assigned_cluster == "ai2/jupiter"
    assert view.budget_clusters(LIMITS) == ("ai2/jupiter",)
    assert view.is_placed
    assert view.placed_at == 5000.0


def test_scheduled_but_not_started_job_is_placed_not_queued():
    # A job Beaker has scheduled is initializing: it holds its slots already and
    # Beaker refuses to raise its priority, so treating it as queued would both
    # double-count the slots and emit a call that fails every pass.
    client = FakeClient([make_job(node_id="node-jup", scheduled=4000, started=0)])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.is_placed
    assert not view.is_queued
    assert view.placed_at == 4000.0


def test_fetch_ignores_jobs_in_other_workspaces():
    client = FakeClient(
        [make_job(job_id="mine"), make_job(job_id="theirs", workspace_id="ws-other")]
    )
    assert [view.id for view in fetch_jobs(client, "ai2/ace")] == ["mine"]


def test_fetch_only_reads_cluster_placement_constraints():
    # A hostname constraint must not be mistaken for a cluster, or a
    # single-cluster job would look multi-cluster and never be managed.
    client = FakeClient(
        [
            make_job(
                clusters=("ai2/jupiter",),
                hostnames=("jupiter-cs-aus-241",),
                env={"CM_PRIORITY": "high"},
            )
        ]
    )
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.clusters == ("ai2/jupiter",)
    assert view.managed_cluster(LIMITS) == "ai2/jupiter"


def test_fetch_reads_a_job_with_no_cluster_constraint():
    client = FakeClient([make_job(clusters=())])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.clusters == ()
    # It could land anywhere, so it is charged to every budget.
    assert view.budget_clusters(LIMITS) == tuple(LIMITS)


def test_fetch_reads_the_replica_group_so_ranks_group_together():
    client = FakeClient(
        [
            make_job(
                job_id=f"rank-{rank}",
                replica_size=4,
                replica_group_id="rg-1",
                env={"CM_PRIORITY": "high"},
            )
            for rank in range(4)
        ]
    )
    views = fetch_jobs(client, "ai2/ace")
    assert {view.replica_group_size for view in views} == {4}
    assert {view.group_key for view in views} == {"rg-1"}
    # A rank is individually manageable; the group decides whether it moves.
    assert all(view.managed_cluster(LIMITS) == "ai2/jupiter" for view in views)


def test_a_lone_job_is_its_own_group():
    client = FakeClient([make_job(job_id="solo", env={"CM_PRIORITY": "high"})])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.group_key == "solo"


def test_fetch_marks_an_interactive_session():
    # A session carries an environment id; a batch job does not.
    client = FakeClient(
        [make_job(job_id="session", environment_id="env-1"), make_job(job_id="batch")]
    )
    sessions = {view.id: view.is_session for view in fetch_jobs(client, "ai2/ace")}
    assert sessions == {"session": True, "batch": False}


def test_fetch_counts_a_cpu_only_job_as_one_slot():
    client = FakeClient([make_job(gpu_count=0)])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.slots == 1


def test_fetch_leaves_cm_priority_unset_when_absent():
    client = FakeClient([make_job(env={"WANDB_NAME": "x"})])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.cm_priority is None
    assert view.managed_cluster(LIMITS) is None


def test_fetch_reads_multi_cluster_constraints_in_order():
    client = FakeClient([make_job(clusters=("ai2/titan", "ai2/jupiter"))])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.clusters == ("ai2/titan", "ai2/jupiter")


# --- node/cluster resolution ------------------------------------------------


def test_node_cluster_is_org_qualified():
    client = FakeClient([], node_clusters={"node-jup": ("jupiter", "ai2")})
    cache: dict[str, str | None] = {}
    assert _cluster_of_node(client, "node-jup", cache) == "ai2/jupiter"
    assert cache == {"node-jup": "ai2/jupiter"}


def test_unresolvable_node_is_cached_as_a_failure():
    # Without negative caching this would be retried for every job, every pass.
    client = FakeClient([], node_clusters={})
    cache: dict[str, str | None] = {}
    assert _cluster_of_node(client, "gone", cache) is None
    assert _cluster_of_node(client, "gone", cache) is None
    assert client.node_lookups == ["gone"]


def test_a_job_on_an_unresolvable_node_is_charged_to_every_budget():
    client = FakeClient([make_job(node_id="gone", started=5000)], node_clusters={})
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.is_placed
    assert view.assigned_cluster is None
    assert view.budget_clusters(LIMITS) == tuple(LIMITS)


def test_node_cache_is_reused_across_passes():
    jobs = [make_job(node_id="node-jup", started=5000)]
    client = FakeClient(jobs)
    cache: dict[str, str | None] = {}
    fetch_jobs(client, "ai2/ace", cache)
    fetch_jobs(client, "ai2/ace", cache)
    assert client.node_lookups == ["node-jup"]


# --- CM_PRIORITY parsing ----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("low", LOW),
        ("normal", NORMAL),
        ("high", HIGH),
        ("urgent", URGENT),
        ("URGENT", URGENT),
        ("  high  ", HIGH),
    ],
)
def test_cm_priority_values_are_parsed(raw, expected):
    assert parse_cm_priority(raw, "job-1") == expected


@pytest.mark.parametrize("raw", ["", "immediate", "highest", "3", "hgih"])
def test_unusable_cm_priority_is_ignored_with_a_warning(raw, caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_cm_priority(raw, "job-1") is None
    assert "job-1" in caplog.text


def test_a_job_with_an_unusable_cm_priority_is_not_managed():
    client = FakeClient([make_job(env={"CM_PRIORITY": "highest"})])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.cm_priority is None
    assert view.managed_cluster(LIMITS) is None


# --- mutation ---------------------------------------------------------------


def test_set_priority_sends_the_job_id_and_priority():
    client = FakeClient([])
    set_priority(client, "job-7", URGENT)
    assert client.priority_calls == [("job-7", URGENT)]


# --- run_pass ---------------------------------------------------------------


def _labelled(job_id, cm, **kwargs):
    return make_job(job_id=job_id, env={"CM_PRIORITY": cm}, **kwargs)


def test_dry_run_changes_nothing():
    client = FakeClient([_labelled("a", "high"), _labelled("b", "high")])
    applied = run_pass(client, "ai2/ace", LIMITS, dry_run=True)
    assert applied == 0
    assert client.priority_calls == []


def test_run_pass_applies_promotions():
    client = FakeClient([_labelled("a", "high"), _labelled("b", "urgent")])
    applied = run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert applied == 2
    assert sorted(client.priority_calls) == [("a", URGENT), ("b", URGENT)]


def test_a_job_we_cannot_modify_does_not_stop_the_pass(caplog):
    # Fail-soft: a teammate's job we lack permission on must not block ours.
    client = FakeClient(
        [_labelled("theirs", "high", author="yyexela"), _labelled("mine", "high")],
        fail_on={"theirs"},
    )
    with caplog.at_level(logging.WARNING):
        applied = run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert applied == 1
    assert client.priority_calls == [("mine", URGENT)]
    assert "theirs" in caplog.text and "yyexela" in caplog.text


def test_demotions_are_attempted_before_promotions():
    # So a pass that dies partway through is never left over allocation.
    client = FakeClient(
        [
            _labelled(
                "holder",
                "normal",
                priority=pb2.JOB_PRIORITY_URGENT,
                node_id="node-jup",
                started=100,
            ),
            _labelled("contender", "high"),
        ]
    )
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    # The holder was already at urgent when this balancer first saw it, so
    # there is nothing remembered to put it back to and it falls back to high.
    assert client.priority_calls == [("holder", HIGH), ("contender", URGENT)]


def test_run_pass_is_idempotent():
    jobs = [_labelled("a", "high"), _labelled("b", "normal")]
    client = FakeClient(jobs)
    assert run_pass(client, "ai2/ace", LIMITS, dry_run=False) == 2

    for job in jobs:
        job.system_details.priority = pb2.JOB_PRIORITY_URGENT
    client = FakeClient(jobs)
    assert run_pass(client, "ai2/ace", LIMITS, dry_run=False) == 0
    assert client.priority_calls == []


def test_multi_cluster_labelled_jobs_are_reported_once_in_aggregate(caplog):
    client = FakeClient(
        [
            _labelled(f"both-{i}", "urgent", clusters=("ai2/titan", "ai2/jupiter"))
            for i in range(5)
        ]
    )
    with caplog.at_level(logging.WARNING):
        applied = run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert applied == 0
    assert client.priority_calls == []
    # One aggregate line, not one per job: most ace jobs are multi-cluster.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "5 job(s)" in warnings[0].getMessage()


def test_unreclaimable_slots_are_broken_out_per_cluster(caplog):
    # A pass can be stuck above the allocation through no fault of its own. The
    # log has to say so, or it reads as a balancer that is not working.
    client = FakeClient(
        [
            make_job(
                job_id="session",
                environment_id="env-1",
                priority=pb2.JOB_PRIORITY_URGENT,
                gpu_count=8,
                node_id="node-jup",
                started=100,
            ),
            make_job(
                job_id="immediate",
                priority=pb2.JOB_PRIORITY_IMMEDIATE,
                gpu_count=16,
                node_id="node-jup",
                started=100,
            ),
        ],
        node_clusters=NODES,
    )
    with caplog.at_level(logging.INFO):
        run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "ai2/jupiter: 8 in interactive sessions; 16 at immediate priority" in m
        for m in messages
    )
    assert client.priority_calls == []


def test_a_replica_group_left_alone_is_reported(caplog):
    # Each rank is individually fine, so this is invisible in the per-job
    # counts: without its own line the group would be skipped silently.
    ranks = [
        _labelled(f"rank-{i}", "high", replica_size=4, replica_group_id="rg-1")
        for i in range(3)
    ]
    ranks.append(make_job(job_id="rank-3", replica_size=4, replica_group_id="rg-1"))
    client = FakeClient(ranks)
    with caplog.at_level(logging.WARNING):
        run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert client.priority_calls == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("1 replica group(s) left alone" in m and "rg-1" in m for m in warnings)


def test_run_pass_reports_usage_after_applying(caplog):
    client = FakeClient([_labelled("a", "high", gpu_count=8)])
    with caplog.at_level(logging.INFO):
        run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    messages = [r.getMessage() for r in caplog.records]
    assert any("urgent slots before: ai2/jupiter 0/72" in m for m in messages)
    assert any("urgent slots after: ai2/jupiter 8/72" in m for m in messages)


# --- partially applied passes -----------------------------------------------
#
# Ordering demotions first makes any prefix of a pass safe, but a failure skips
# one action and carries on, so what lands is an arbitrary subset. These cover
# the subset that matters: the one where the demotion paying for a grant is the
# call that failed.


def _holder(job_id, cm, cluster="ai2/jupiter", **kwargs):
    """A placed job holding urgent, i.e. one the balancer can demote."""
    return _labelled(
        job_id,
        cm,
        priority=pb2.JOB_PRIORITY_URGENT,
        clusters=(cluster,),
        node_id="node-jup" if cluster == "ai2/jupiter" else "node-tit",
        started=100,
        **kwargs,
    )


NODES = {"node-jup": ("jupiter", "ai2"), "node-tit": ("titan", "ai2")}


def test_a_refused_demotion_defers_the_grant_it_was_paying_for(caplog):
    # The whole point: without this the pass ends at 16/8, and says so.
    client = FakeClient(
        [
            _holder("theirs", "normal", author="yyexela"),
            _labelled("mine", "high"),
        ],
        node_clusters=NODES,
        fail_on={"theirs"},
    )
    with caplog.at_level(logging.INFO):
        applied = run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    assert applied == 0
    assert client.priority_calls == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("urgent slots after: ai2/jupiter 8/8" in m for m in messages)
    assert any("not granting urgent to mine" in m for m in messages)


def test_a_refused_demotion_on_one_cluster_does_not_stall_the_other():
    client = FakeClient(
        [
            _holder("theirs", "normal", author="yyexela"),
            _labelled("mine", "high"),
            _labelled("elsewhere", "high", clusters=("ai2/titan",)),
        ],
        node_clusters=NODES,
        fail_on={"theirs"},
    )
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8, "ai2/titan": 8}, dry_run=False)
    assert client.priority_calls == [("elsewhere", URGENT)]


def test_a_deferred_grant_is_retried_once_the_demotion_lands():
    jobs = [_holder("theirs", "normal", author="yyexela"), _labelled("mine", "high")]
    client = FakeClient(jobs, node_clusters=NODES, fail_on={"theirs"})
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    assert client.priority_calls == []

    # Next pass, with permission no longer refused: both halves go through.
    client = FakeClient(jobs, node_clusters=NODES)
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    assert client.priority_calls == [("theirs", HIGH), ("mine", URGENT)]


def test_only_the_grant_the_deficit_pays_for_is_deferred():
    # An 8-slot demotion refused must not defer 32 slots' worth of grants.
    client = FakeClient(
        [
            _holder("theirs", "normal", author="yyexela", gpu_count=8),
            _labelled("small", "high", gpu_count=8),
            _labelled("also", "high", gpu_count=8),
        ],
        node_clusters=NODES,
        fail_on={"theirs"},
    )
    run_pass(client, "ai2/ace", {"ai2/jupiter": 16}, dry_run=False)
    # 16 - 8 held by the refusal leaves room for exactly one of the two.
    assert client.priority_calls == [("also", URGENT)]


def test_a_refused_rank_stops_the_rest_of_its_group_being_granted():
    """A group that cannot be granted whole stops rather than spending more.

    The ranks granted before the refusal keep urgent: the next pass re-decides
    from what is really there and either finishes the group or takes it back,
    which is cheaper than a rollback that can fail in its turn. Permission is
    per owner and every rank shares one, so in practice the first rank fails
    and nothing is spent at all.
    """
    ranks = [
        _labelled(f"rank-{i}", "high", replica_size=4, replica_group_id="rg-1")
        for i in range(4)
    ]
    client = FakeClient(ranks, node_clusters=NODES, fail_on={"rank-2"})
    run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert [job_id for job_id, _ in client.priority_calls] == ["rank-0", "rank-1"]


def test_a_replica_group_grant_is_deferred_as_a_unit():
    client = FakeClient(
        [
            _holder("theirs", "normal", author="yyexela", gpu_count=8),
            *[
                _labelled(
                    f"rank-{i}",
                    "high",
                    gpu_count=8,
                    replica_size=2,
                    replica_group_id="rg-1",
                )
                for i in range(2)
            ],
        ],
        node_clusters=NODES,
        fail_on={"theirs"},
    )
    # 16 for the group leaves nothing for the holder, so the holder must be
    # demoted to pay for it -- and that is the call that is refused.
    run_pass(client, "ai2/ace", {"ai2/jupiter": 16}, dry_run=False)
    assert client.priority_calls == []


def test_a_deferred_group_repays_the_deficit_with_all_of_its_slots():
    # A group is deferred whole, so it stops spending everything it was going to
    # spend -- not just the one rank that reached the check. Deferring further
    # grants after that would strand slots the refusal did not actually hold.
    client = FakeClient(
        [
            _holder("theirs", "normal", author="yyexela", gpu_count=16),
            *[
                _labelled(
                    f"rank-{i}",
                    "high",
                    gpu_count=8,
                    replica_size=4,
                    replica_group_id="rg-1",
                    created=100,
                )
                for i in range(4)
            ],
            _labelled("lone", "high", gpu_count=8, created=200),
        ],
        node_clusters=NODES,
        fail_on={"theirs"},
    )
    # 32 for the group and 8 for the lone job fill the 40, so the 16-slot holder
    # must be demoted to pay for them -- and that is the call that is refused.
    run_pass(client, "ai2/ace", {"ai2/jupiter": 40}, dry_run=False)
    # Abandoning the group frees 32 against a 16-slot deficit, so the lone job
    # still fits and must not be deferred with it.
    assert client.priority_calls == [("lone", URGENT)]


def test_negative_node_cache_is_dropped_between_passes():
    # A transient error must not charge a job to every budget indefinitely.
    jobs = [make_job(node_id="flaky", started=5000)]
    client = FakeClient(jobs)
    cache: dict[str, str | None] = {}
    (view,) = fetch_jobs(client, "ai2/ace", cache)
    assert view.assigned_cluster is None
    assert view.budget_clusters(LIMITS) == tuple(LIMITS)

    client._node_clusters["flaky"] = ("jupiter", "ai2")
    (view,) = fetch_jobs(client, "ai2/ace", cache)
    assert view.assigned_cluster == "ai2/jupiter"


@pytest.mark.parametrize("seed", range(200))
def test_a_pass_never_ends_over_allocation_however_calls_fail(seed):
    """Any subset of a pass landing must leave us no worse than we started.

    Every job here is manageable and single-cluster, so nothing unreclaimable
    is propping the usage up: an overshoot could only be the balancer's doing.
    """
    rng = random.Random(seed)
    limits = {"ai2/jupiter": 32, "ai2/titan": 16}
    jobs = []
    for i in range(rng.randrange(1, 10)):
        cluster = rng.choice(list(limits))
        placed = rng.random() < 0.5
        jobs.append(
            _labelled(
                f"job-{i}",
                rng.choice(["low", "normal", "high", "urgent"]),
                priority=rng.choice(
                    [
                        pb2.JOB_PRIORITY_LOW,
                        pb2.JOB_PRIORITY_NORMAL,
                        pb2.JOB_PRIORITY_HIGH,
                        pb2.JOB_PRIORITY_URGENT,
                    ]
                ),
                gpu_count=rng.choice([1, 8, 16]),
                clusters=(cluster,),
                node_id=("node-jup" if cluster == "ai2/jupiter" else "node-tit")
                if placed
                else "",
                started=100 + i if placed else 0,
                created=100 + i,
            )
        )
    fail_on = {job.id for job in jobs if rng.random() < 0.5}

    client = FakeClient(jobs, node_clusters=NODES, fail_on=fail_on)
    before = fetch_jobs(client, "ai2/ace")
    run_pass(client, "ai2/ace", limits, dry_run=False)

    landed = dict(client.priority_calls)
    after = [
        replace(view, priority=landed[view.id]) if view.id in landed else view
        for view in before
    ]
    start = urgent_usage(before, limits)
    end = urgent_usage(after, limits)
    for cluster, limit in limits.items():
        assert end[cluster] <= max(
            limit, start[cluster]
        ), f"{cluster}: {start[cluster]} -> {end[cluster]}, limit {limit}"


# --- configuration ----------------------------------------------------------


def test_limits_default_to_the_team_allocation():
    assert parse_limits(None) == {"ai2/jupiter": 72, "ai2/titan": 32}


def test_an_override_merges_rather_than_replacing():
    # Replacing would silently unmanage jupiter entirely.
    assert parse_limits(["ai2/titan=0"]) == {"ai2/jupiter": 72, "ai2/titan": 0}


def test_limits_can_add_a_new_cluster():
    limits = parse_limits(["ai2/saturn=16"])
    assert limits["ai2/saturn"] == 16
    assert limits["ai2/jupiter"] == 72


@pytest.mark.parametrize(
    "bad", ["ai2/jupiter", "ai2/jupiter=", "ai2/jupiter=lots", "=8", "ai2/jupiter=-8"]
)
def test_a_malformed_limit_is_rejected(bad):
    # Silently ignoring this would run the balancer against a wrong allocation.
    with pytest.raises(ValueError):
        parse_limits([bad])


def test_an_unknown_cluster_name_is_rejected_at_startup():
    # A typo would manage nothing and look exactly like a quiet cluster.
    client = FakeClient([], unknown_clusters={"ai2/jupiter-cirrascale-2"})
    with pytest.raises(SystemExit):
        validate_limits(client, {"ai2/jupiter-cirrascale-2": 72})


def test_valid_cluster_names_pass_validation():
    validate_limits(FakeClient([]), LIMITS)


# --- the entrypoint ---------------------------------------------------------


class _Stop(Exception):
    """Breaks out of the --interval loop, which otherwise never returns."""


def _with_client(monkeypatch, client):
    monkeypatch.setattr(balance, "Beaker", SimpleNamespace(from_env=lambda: client))


def test_one_pass_is_run_and_returned_from_by_default(monkeypatch):
    client = FakeClient([_labelled("a", "high")])
    _with_client(monkeypatch, client)
    monkeypatch.setattr(balance.time, "sleep", lambda seconds: pytest.fail("looped"))
    assert balance.main(["--dry-run"]) == 0
    assert client.priority_calls == []


def test_a_malformed_limit_exits_before_touching_beaker(monkeypatch):
    monkeypatch.setattr(
        balance, "Beaker", SimpleNamespace(from_env=lambda: pytest.fail("connected"))
    )
    with pytest.raises(SystemExit):
        balance.main(["--limit", "ai2/jupiter"])


def test_a_failed_pass_is_not_swallowed_without_an_interval(monkeypatch):
    # A one-shot run must report failure rather than exit 0 having done nothing.
    _with_client(monkeypatch, FakeClient([]))
    monkeypatch.setattr(
        balance, "run_pass", lambda *a, **k: (_ for _ in ()).throw(BeakerError("down"))
    )
    with pytest.raises(BeakerError):
        balance.main([])


def test_a_transient_beaker_error_does_not_kill_an_interval_loop(monkeypatch, caplog):
    # The failure mode this guards is a cron balancer that quietly stops
    # balancing after one bad poll.
    _with_client(monkeypatch, FakeClient([]))
    passes = []

    def flaky(*args, **kwargs):
        passes.append(1)
        if len(passes) == 1:
            raise BeakerError("beaker is down")
        return 0

    monkeypatch.setattr(balance, "run_pass", flaky)

    def sleeper(seconds):
        assert seconds == 5.0
        if len(passes) >= 2:
            raise _Stop

    monkeypatch.setattr(balance.time, "sleep", sleeper)
    with caplog.at_level(logging.ERROR), pytest.raises(_Stop):
        balance.main(["--interval", "5"])
    assert len(passes) == 2
    assert "pass failed" in caplog.text


# --- remembering where a job rests, across passes ---------------------------


def test_a_job_is_returned_to_the_priority_it_was_raised_from(state_in_tmp_path):
    """The point of the memory: give the slot back, leave everything else.

    The first pass raises a ``normal`` job to urgent. By the second the
    allocation has gone, and the job must land back on ``normal`` -- where its
    owner submitted it -- rather than on the ``high`` fallback.
    """
    submitted = _labelled("mine", "high", priority=pb2.JOB_PRIORITY_NORMAL)
    client = FakeClient([submitted])
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    assert client.priority_calls == [("mine", URGENT)]

    raised = _labelled(
        "mine",
        "high",
        priority=pb2.JOB_PRIORITY_URGENT,
        node_id="node-jup",
        started=100,
    )
    client = FakeClient([raised])
    run_pass(client, "ai2/ace", {"ai2/jupiter": 0}, dry_run=False)
    assert client.priority_calls == [("mine", NORMAL)]


def test_a_manual_change_is_adopted_as_the_new_resting_priority(state_in_tmp_path):
    # The owner dropped the job to low between passes. That is now where it
    # belongs, overriding the normal recorded when the balancer first saw it.
    client = FakeClient([_labelled("mine", "high", priority=pb2.JOB_PRIORITY_NORMAL)])
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)

    client = FakeClient([_labelled("mine", "high", priority=pb2.JOB_PRIORITY_LOW)])
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    assert client.priority_calls == [("mine", URGENT)]

    raised = _holder("mine", "high")
    client = FakeClient([raised], node_clusters=NODES)
    run_pass(client, "ai2/ace", {"ai2/jupiter": 0}, dry_run=False)
    assert client.priority_calls == [("mine", LOW)]


def test_a_requeued_job_is_not_ratcheted_up_by_preemption(state_in_tmp_path):
    """A preempted job comes back at the priority the balancer set, not its own.

    Beaker gives the retry a new id, so without following ``retry_ancestor_id``
    every preemption would lose the submitted priority and leave the job at the
    ``high`` fallback -- ratcheting the whole workspace up over time.
    """
    client = FakeClient([_labelled("first", "high", priority=pb2.JOB_PRIORITY_LOW)])
    run_pass(client, "ai2/ace", {"ai2/jupiter": 8}, dry_run=False)
    assert client.priority_calls == [("first", URGENT)]

    # Preempted, requeued under a new id, handed back at urgent by the source.
    retry = _labelled(
        "second", "high", priority=pb2.JOB_PRIORITY_URGENT, retry_ancestor_id="first"
    )
    client = FakeClient([retry])
    run_pass(client, "ai2/ace", {"ai2/jupiter": 0}, dry_run=False)
    assert client.priority_calls == [("second", LOW)]


def test_a_finished_job_is_dropped_from_the_state_file(state_in_tmp_path):
    client = FakeClient([_labelled("mine", "high", priority=pb2.JOB_PRIORITY_NORMAL)])
    run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert "mine" in json.loads(state_in_tmp_path.read_text())["resting"]

    run_pass(FakeClient([]), "ai2/ace", LIMITS, dry_run=False)
    assert json.loads(state_in_tmp_path.read_text())["resting"] == {}


def test_a_dry_run_does_not_write_the_state_file(state_in_tmp_path):
    # Reporting a pass must not change what a real one would then do.
    client = FakeClient([_labelled("mine", "high", priority=pb2.JOB_PRIORITY_NORMAL)])
    run_pass(client, "ai2/ace", LIMITS, dry_run=True)
    assert not state_in_tmp_path.exists()


def test_an_unreadable_state_file_does_not_stop_a_pass(state_in_tmp_path, caplog):
    # Forgetting costs the high fallback; refusing to run costs the allocation.
    state_in_tmp_path.write_text("{not json")
    client = FakeClient([_holder("mine", "high")], node_clusters=NODES)
    with caplog.at_level(logging.WARNING):
        run_pass(client, "ai2/ace", {"ai2/jupiter": 0}, dry_run=False)
    assert client.priority_calls == [("mine", HIGH)]
    assert "could not read" in caplog.text


def test_an_unwritable_state_file_does_not_stop_a_pass(state_in_tmp_path, caplog):
    state_in_tmp_path.parent.joinpath("resting.json").mkdir()
    client = FakeClient([_labelled("mine", "high")])
    with caplog.at_level(logging.WARNING):
        applied = run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert applied == 1
    assert "could not write" in caplog.text


def test_state_from_a_future_version_is_ignored(state_in_tmp_path, caplog):
    state_in_tmp_path.write_text(json.dumps({"version": 99, "resting": {"mine": LOW}}))
    client = FakeClient([_holder("mine", "high")], node_clusters=NODES)
    with caplog.at_level(logging.WARNING):
        run_pass(client, "ai2/ace", {"ai2/jupiter": 0}, dry_run=False)
    assert client.priority_calls == [("mine", HIGH)]


def test_a_nonsense_priority_in_the_state_file_is_ignored(state_in_tmp_path):
    # Never restore a job to urgent or immediate, whatever the file claims.
    state_in_tmp_path.write_text(
        json.dumps({"version": 1, "resting": {"mine": URGENT, "other": 99}})
    )
    client = FakeClient([_holder("mine", "high")], node_clusters=NODES)
    run_pass(client, "ai2/ace", {"ai2/jupiter": 0}, dry_run=False)
    assert client.priority_calls == [("mine", HIGH)]


def test_the_state_is_written_before_any_change_is_applied(
    state_in_tmp_path, monkeypatch
):
    """A pass that dies midway must still have recorded what it saw.

    The record is what lets the next pass put the job back; taking it after the
    fact would read the priority this pass had just imposed.
    """
    client = FakeClient([_labelled("mine", "high", priority=pb2.JOB_PRIORITY_NORMAL)])
    seen = {}

    def crash(*args, **kwargs):
        seen.update(json.loads(state_in_tmp_path.read_text())["resting"])
        raise BeakerError("died midway")

    monkeypatch.setattr(client.job, "rpc_request", crash)
    run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert seen == {"mine": NORMAL}


def test_the_state_path_is_taken_from_the_command_line(tmp_path, monkeypatch):
    chosen = tmp_path / "elsewhere" / "resting.json"
    client = FakeClient([_labelled("mine", "high", priority=pb2.JOB_PRIORITY_NORMAL)])
    _with_client(monkeypatch, client)
    assert balance.main(["--state", str(chosen)]) == 0
    assert json.loads(chosen.read_text())["resting"] == {"mine": NORMAL}
