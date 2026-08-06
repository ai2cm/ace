"""Integration tests for the balancer's Beaker-facing layer.

These build real protobuf messages of the shapes Beaker returns and drive
``fetch_jobs``, ``set_priority`` and ``run_pass`` against a fake client, so the
parsing and mutation paths are covered without touching a live workspace.
"""

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("beaker", reason="beaker-py is not a core fme dependency")

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
    validate_limits,
)
from beaker import beaker_pb2 as pb2  # noqa: E402
from beaker.exceptions import BeakerError  # noqa: E402

WORKSPACE_ID = "ws-ace"
LIMITS = {"ai2/jupiter": 72, "ai2/titan": 32}


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
    workspace_id=WORKSPACE_ID,
):
    """Build a pb2.Job mirroring the shape Beaker returns for a live job."""
    job = pb2.Job(
        id=job_id,
        name=name,
        author_reference=author,
        workspace_id=workspace_id,
        task_id="task-1",
        workload_id="workload-1",
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


def test_fetch_reads_replica_group_size():
    client = FakeClient([make_job(replica_size=4, env={"CM_PRIORITY": "high"})])
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.replica_group_size == 4
    assert view.managed_cluster(LIMITS) is None


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
    assert client.priority_calls == [("holder", NORMAL), ("contender", URGENT)]


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


def test_run_pass_reports_usage_after_applying(caplog):
    client = FakeClient([_labelled("a", "high", gpu_count=8)])
    with caplog.at_level(logging.INFO):
        run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    messages = [r.getMessage() for r in caplog.records]
    assert any("urgent slots before: ai2/jupiter 0/72" in m for m in messages)
    assert any("urgent slots after: ai2/jupiter 8/72" in m for m in messages)


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
    "bad", ["ai2/jupiter", "ai2/jupiter=", "ai2/jupiter=lots", "=8"]
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
