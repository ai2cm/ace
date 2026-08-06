"""Tests for the balancer's decision logic.

These exercise ``decide`` against constructed job lists, so no Beaker calls are
made.
"""

import itertools
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
from balance import (
    HIGH,
    LOW,
    NORMAL,
    URGENT,
    Action,
    JobView,
    _cluster_of_node,
    decide,
    resting_priority,
)

LIMITS = {"ai2/jupiter": 72, "ai2/titan": 32}

_ids = itertools.count()


def job(
    priority=HIGH,
    cm_priority=HIGH,
    slots=8,
    clusters=("ai2/jupiter",),
    assigned_cluster=None,
    queued_at=0.0,
    started_at=None,
    name="job",
    author="jeremym",
):
    """Build a JobView, defaulting to a queued 8-GPU jupiter job."""
    return JobView(
        id=f"job-{next(_ids)}",
        name=name,
        author=author,
        priority=priority,
        slots=slots,
        clusters=clusters,
        assigned_cluster=assigned_cluster,
        cm_priority=cm_priority,
        queued_at=queued_at,
        started_at=started_at,
    )


def running(**kwargs):
    """A job that has landed on a node, and so cannot be promoted."""
    kwargs.setdefault("started_at", 100.0)
    kwargs.setdefault("assigned_cluster", kwargs.get("clusters", ("ai2/jupiter",))[0])
    return job(**kwargs)


def changes(actions: list[Action]) -> dict[str, int]:
    return {action.job.id: action.to_priority for action in actions}


def test_resting_priority_drops_urgent_to_high():
    assert resting_priority(URGENT) == HIGH
    assert resting_priority(HIGH) == HIGH
    assert resting_priority(LOW) == LOW


def test_queued_jobs_are_promoted_to_fill_the_allocation():
    jobs = [job(priority=HIGH, cm_priority=HIGH) for _ in range(3)]
    assert changes(decide(jobs, LIMITS)) == {j.id: URGENT for j in jobs}


def test_promotion_stops_at_the_allocation():
    # 10 x 8 slots = 80, against a 72 slot allocation.
    jobs = [job(queued_at=float(i)) for i in range(10)]
    promoted = changes(decide(jobs, LIMITS))
    assert len(promoted) == 9
    # The longest-waiting jobs are promoted first, so the newest misses out.
    assert jobs[-1].id not in promoted


def test_higher_cm_priority_is_granted_first():
    low = job(priority=LOW, cm_priority=LOW, slots=72, queued_at=0.0)
    high = job(cm_priority=HIGH, slots=72, queued_at=10.0)
    result = changes(decide([low, high], LIMITS))
    # Only one fits. The high-priority job wins despite waiting less.
    assert result == {high.id: URGENT}


def test_job_is_moved_to_its_resting_priority_when_not_granted():
    # "own it fully": a labelled job sits at its own label, whatever it was
    # submitted at, once it is clear it will not get an urgent slot.
    mislabelled = job(priority=HIGH, cm_priority=LOW, slots=8)
    hog = job(priority=URGENT, cm_priority=None, slots=72)
    assert changes(decide([hog, mislabelled], LIMITS)) == {mislabelled.id: LOW}


def test_over_allocation_demotes_lowest_cm_priority_first():
    keep = running(priority=URGENT, cm_priority=HIGH, slots=64)
    drop = running(priority=URGENT, cm_priority=LOW, slots=8)
    result = changes(decide([keep, drop], {"ai2/jupiter": 64}))
    assert result == {drop.id: LOW}


def test_queued_jobs_are_demoted_before_running_ones_at_the_same_level():
    still_queued = job(priority=URGENT, cm_priority=HIGH, slots=8)
    on_gpu = running(priority=URGENT, cm_priority=HIGH, slots=8)
    result = changes(decide([still_queued, on_gpu], {"ai2/jupiter": 8}))
    assert result == {still_queued.id: HIGH}


def test_longest_running_job_is_demoted_first():
    old = running(priority=URGENT, cm_priority=HIGH, slots=8, started_at=100.0)
    new = running(priority=URGENT, cm_priority=HIGH, slots=8, started_at=900.0)
    result = changes(decide([old, new], {"ai2/jupiter": 8}))
    assert result == {old.id: HIGH}


def test_longest_queued_job_is_promoted_first():
    old = job(cm_priority=HIGH, slots=8, queued_at=100.0)
    new = job(cm_priority=HIGH, slots=8, queued_at=900.0)
    result = changes(decide([old, new], {"ai2/jupiter": 8}))
    assert result == {old.id: URGENT}


def test_running_job_is_never_promoted():
    # Beaker rejects raising the priority of a running job.
    on_gpu = running(priority=HIGH, cm_priority=URGENT)
    assert decide([on_gpu], LIMITS) == []


def test_running_job_below_its_resting_priority_is_left_alone():
    on_gpu = running(priority=NORMAL, cm_priority=HIGH)
    assert decide([on_gpu], LIMITS) == []


def test_running_non_urgent_job_does_not_displace_others():
    # The urgent-labelled running job cannot be promoted, so it must not push
    # the queued job out of the allocation.
    blocked = running(priority=HIGH, cm_priority=URGENT, slots=8)
    queued = job(cm_priority=NORMAL, slots=8)
    result = changes(decide([blocked, queued], {"ai2/jupiter": 8}))
    assert result == {queued.id: URGENT}


def test_multi_cluster_jobs_are_never_modified():
    both = job(clusters=("ai2/titan", "ai2/jupiter"), cm_priority=URGENT)
    assert decide([both], LIMITS) == []


def test_queued_multi_cluster_urgent_job_counts_against_every_cluster():
    both = job(priority=URGENT, cm_priority=None, clusters=("ai2/titan", "ai2/jupiter"))
    on_jupiter = job(cm_priority=HIGH, slots=72, clusters=("ai2/jupiter",))
    on_titan = job(cm_priority=HIGH, slots=32, clusters=("ai2/titan",))
    # The multi-cluster job takes 8 slots from both budgets, so neither of the
    # exactly-sized jobs fits any more.
    assert decide([both, on_jupiter, on_titan], LIMITS) == []


def test_assigned_multi_cluster_job_counts_only_where_it_landed():
    landed = running(
        priority=URGENT,
        cm_priority=None,
        clusters=("ai2/titan", "ai2/jupiter"),
        assigned_cluster="ai2/titan",
    )
    on_jupiter = job(cm_priority=HIGH, slots=72, clusters=("ai2/jupiter",))
    result = changes(decide([landed, on_jupiter], LIMITS))
    assert result == {on_jupiter.id: URGENT}


def test_jobs_without_cm_priority_count_but_are_not_modified():
    unlabelled = job(priority=URGENT, cm_priority=None, slots=72)
    labelled = job(cm_priority=HIGH, slots=8)
    # The whole allocation is spoken for by a job we may not touch.
    assert decide([unlabelled, labelled], LIMITS) == []


def test_first_fit_skips_a_job_that_does_not_fit():
    big = job(cm_priority=HIGH, slots=16, queued_at=0.0)
    small = job(cm_priority=HIGH, slots=8, queued_at=10.0)
    result = changes(decide([big, small], {"ai2/jupiter": 8}))
    assert result == {small.id: URGENT}


def test_a_blocked_level_does_not_reserve_slots_for_itself():
    # The high job cannot fit, so normal jobs behind it still get the slots.
    big = job(cm_priority=HIGH, slots=16)
    small = job(cm_priority=NORMAL, slots=8)
    result = changes(decide([big, small], {"ai2/jupiter": 8}))
    assert result == {small.id: URGENT}


def test_higher_level_job_displaces_lower_level_urgent_jobs():
    # The rebalancing case: normal-priority jobs holding urgent are dropped so
    # a high-priority job can take the slots.
    holders = [running(priority=URGENT, cm_priority=NORMAL, slots=4) for _ in range(2)]
    contender = job(cm_priority=HIGH, slots=8)
    result = changes(decide([*holders, contender], {"ai2/jupiter": 8}))
    assert result == {
        holders[0].id: NORMAL,
        holders[1].id: NORMAL,
        contender.id: URGENT,
    }


def test_cluster_allocations_are_independent():
    jupiter = job(clusters=("ai2/jupiter",), cm_priority=HIGH, slots=72)
    titan = job(clusters=("ai2/titan",), cm_priority=HIGH, slots=32)
    result = changes(decide([jupiter, titan], LIMITS))
    assert result == {jupiter.id: URGENT, titan.id: URGENT}


def test_jobs_on_unbudgeted_clusters_are_left_alone():
    saturn = job(clusters=("ai2/saturn",), cm_priority=URGENT, priority=URGENT)
    assert decide([saturn], LIMITS) == []


@pytest.mark.parametrize("cm_priority", [LOW, NORMAL, HIGH, URGENT])
def test_any_labelled_job_can_be_granted_an_urgent_slot(cm_priority):
    queued = job(cm_priority=cm_priority)
    assert changes(decide([queued], LIMITS)) == {queued.id: URGENT}


def test_demoted_job_returns_to_its_own_label_not_to_high():
    over = running(priority=URGENT, cm_priority=LOW, slots=8)
    assert changes(decide([over], {"ai2/jupiter": 0})) == {over.id: LOW}


def apply(jobs: list[JobView], actions: list[Action]) -> list[JobView]:
    """Return the job list as it would be after the actions are applied."""
    new_priority = {action.job.id: action.to_priority for action in actions}
    return [
        replace(j, priority=new_priority[j.id]) if j.id in new_priority else j
        for j in jobs
    ]


def usage(jobs: list[JobView], limits: dict[str, int]) -> dict[str, int]:
    used = dict.fromkeys(limits, 0)
    for job in jobs:
        if job.priority == URGENT:
            for cluster in job.budget_clusters(limits):
                used[cluster] += job.slots
    return used


def random_population(rng: random.Random) -> list[JobView]:
    jobs = []
    for _ in range(rng.randint(0, 40)):
        is_running = rng.random() < 0.5
        clusters = rng.choice(
            [
                ("ai2/jupiter",),
                ("ai2/titan",),
                ("ai2/titan", "ai2/jupiter"),
                ("ai2/saturn",),
            ]
        )
        jobs.append(
            job(
                priority=rng.choice([LOW, NORMAL, HIGH, URGENT]),
                cm_priority=rng.choice([None, LOW, NORMAL, HIGH, URGENT]),
                slots=rng.choice([1, 2, 4, 8, 16, 40]),
                clusters=clusters,
                assigned_cluster=rng.choice(clusters) if is_running else None,
                started_at=float(rng.randint(1, 10_000)) if is_running else None,
                queued_at=float(rng.randint(1, 10_000)),
            )
        )
    return jobs


@pytest.mark.parametrize("seed", range(300))
def test_allocation_is_never_exceeded_by_our_own_doing(seed):
    """The core guarantee: a pass never pushes usage above the allocation.

    Usage may start over the limit because of jobs the balancer cannot modify.
    In that case it must not make matters worse.
    """
    rng = random.Random(seed)
    jobs = random_population(rng)
    before = usage(jobs, LIMITS)
    after = usage(apply(jobs, decide(jobs, LIMITS)), LIMITS)
    for cluster, limit in LIMITS.items():
        assert after[cluster] <= max(
            limit, before[cluster]
        ), f"{cluster}: {before[cluster]} -> {after[cluster]}, limit {limit}"


@pytest.mark.parametrize("seed", range(300))
def test_a_pass_always_converges(seed):
    """A second pass must be a no-op, or the cron would flap jobs forever."""
    rng = random.Random(seed)
    jobs = random_population(rng)
    settled = apply(jobs, decide(jobs, LIMITS))
    assert decide(settled, LIMITS) == []


@pytest.mark.parametrize("seed", range(300))
def test_only_labelled_single_cluster_jobs_are_ever_modified(seed):
    rng = random.Random(seed)
    jobs = random_population(rng)
    for action in decide(jobs, LIMITS):
        assert action.job.cm_priority is not None
        assert action.job.managed_cluster(LIMITS) is not None
        # Beaker refuses to raise a running job; we must never try.
        if not action.job.is_queued:
            assert action.to_priority < action.from_priority


@pytest.mark.parametrize("seed", range(300))
def test_a_job_is_never_left_above_urgent_or_below_its_label(seed):
    rng = random.Random(seed)
    jobs = random_population(rng)
    for action in decide(jobs, LIMITS):
        cm_priority = action.job.cm_priority
        assert cm_priority is not None
        assert action.to_priority in (URGENT, resting_priority(cm_priority))


def test_node_cluster_is_org_qualified():
    # Cluster.name is bare ("jupiter") but placement constraints and our
    # allocation are qualified ("ai2/jupiter"). If these are not reconciled,
    # assigned jobs match no budget and their urgent slots go uncounted.
    client = SimpleNamespace(
        node=SimpleNamespace(get=lambda _: SimpleNamespace(cluster_id="c1")),
        cluster=SimpleNamespace(
            get=lambda _: SimpleNamespace(name="jupiter", organization_name="ai2")
        ),
    )
    cache: dict[str, str] = {}
    assert _cluster_of_node(client, "node-1", cache) == "ai2/jupiter"
    assert cache == {"node-1": "ai2/jupiter"}
