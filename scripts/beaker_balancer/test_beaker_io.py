"""Integration tests for the balancer's Beaker-facing layer.

These build real protobuf messages of the shapes Beaker returns and drive
``fetch_jobs``, ``set_priority`` and ``run_pass`` against a fake client, so the
parsing and mutation paths are covered without touching a live workspace.
"""

import logging
from types import SimpleNamespace

import pytest
from balance import (
    HIGH,
    LOW,
    NORMAL,
    URGENT,
    fetch_jobs,
    parse_cm_priority,
    parse_limits,
    run_pass,
    set_priority,
)
from beaker import beaker_pb2 as pb2
from beaker.exceptions import BeakerError

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
    started=0,
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

    job.status.created.seconds = created
    if started:
        job.status.started.seconds = started
        job.status.status = pb2.STATUS_RUNNING
    else:
        job.status.status = pb2.STATUS_QUEUED
    if node_id:
        job.assignment_details.node_id = node_id
    return job


class FakeClient:
    """Stands in for beaker.Beaker over the surface the balancer uses."""

    def __init__(self, jobs, node_clusters=None, fail_on=()):
        self._jobs = list(jobs)
        # node id -> (bare cluster name, org name)
        self._node_clusters = node_clusters or {"node-jup": ("jupiter", "ai2")}
        self._fail_on = set(fail_on)
        self.priority_calls: list[tuple[str, int]] = []

        outer = self

        class _JobService:
            service = SimpleNamespace(UpdateJobSourcePriority=object())

            def list(self, *, finalized=None, **kwargs):
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
        self.node = SimpleNamespace(
            get=lambda node_id: SimpleNamespace(cluster_id=node_id)
        )
        self.cluster = SimpleNamespace(
            get=lambda cluster_id: SimpleNamespace(
                name=outer._node_clusters[cluster_id][0],
                organization_name=outer._node_clusters[cluster_id][1],
            )
        )


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
    assert view.started_at is None


def test_fetch_qualifies_the_assigned_cluster_with_its_org():
    # Regression: Cluster.name is bare, so an unqualified value would match no
    # budget and the job's urgent slots would go uncounted.
    client = FakeClient(
        [make_job(node_id="node-jup", started=5000)],
        node_clusters={"node-jup": ("jupiter", "ai2")},
    )
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.assigned_cluster == "ai2/jupiter"
    assert view.budget_clusters(LIMITS) == ("ai2/jupiter",)
    assert not view.is_queued
    assert view.started_at == 5000.0


def test_fetch_ignores_jobs_in_other_workspaces():
    client = FakeClient(
        [make_job(job_id="mine"), make_job(job_id="theirs", workspace_id="ws-other")]
    )
    assert [view.id for view in fetch_jobs(client, "ai2/ace")] == ["mine"]


def test_fetch_only_reads_cluster_placement_constraints():
    # A hostname constraint must not be mistaken for a cluster, or a
    # single-cluster job would look multi-cluster and never be managed.
    client = FakeClient(
        [make_job(clusters=("ai2/jupiter",), hostnames=("jupiter-cs-aus-241",))]
    )
    (view,) = fetch_jobs(client, "ai2/ace")
    assert view.clusters == ("ai2/jupiter",)
    assert view.managed_cluster(LIMITS) is None  # no CM_PRIORITY set


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


def test_run_pass_is_idempotent():
    # A second pass over the resulting state must be a no-op, or the balancer
    # would flap jobs between priorities on every cron tick.
    jobs = [_labelled("a", "high"), _labelled("b", "normal")]
    client = FakeClient(jobs)
    assert run_pass(client, "ai2/ace", LIMITS, dry_run=False) == 2

    for job in jobs:
        job.system_details.priority = URGENT
    client = FakeClient(jobs)
    assert run_pass(client, "ai2/ace", LIMITS, dry_run=False) == 0
    assert client.priority_calls == []


def test_multi_cluster_labelled_job_is_reported_but_untouched(caplog):
    client = FakeClient(
        [_labelled("both", "urgent", clusters=("ai2/titan", "ai2/jupiter"))]
    )
    with caplog.at_level(logging.WARNING):
        applied = run_pass(client, "ai2/ace", LIMITS, dry_run=False)
    assert applied == 0
    assert client.priority_calls == []
    assert "both" in caplog.text


# --- configuration ----------------------------------------------------------


def test_limits_default_to_the_team_allocation():
    assert parse_limits(None) == {"ai2/jupiter": 72, "ai2/titan": 32}


def test_limits_can_be_overridden():
    assert parse_limits(["ai2/jupiter=8", "ai2/titan=4"]) == {
        "ai2/jupiter": 8,
        "ai2/titan": 4,
    }


@pytest.mark.parametrize("bad", ["ai2/jupiter", "ai2/jupiter=", "ai2/jupiter=lots"])
def test_a_malformed_limit_is_rejected(bad):
    # Silently ignoring this would run the balancer against a wrong allocation.
    with pytest.raises(ValueError):
        parse_limits([bad])
