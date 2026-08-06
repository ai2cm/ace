"""Tests for the balancer's decision logic.

These exercise ``decide`` against constructed job lists, so no Beaker calls are
made. The Beaker-facing layer is covered in ``test_beaker_io.py``.
"""

import itertools
import random
from dataclasses import replace

import pytest

pytest.importorskip("beaker", reason="beaker-py is not a core fme dependency")

from balance import (  # noqa: E402
    HIGH,
    LOW,
    NORMAL,
    URGENT,
    Action,
    JobView,
    decide,
    resting_priority,
)

LIMITS = {"ai2/jupiter": 72, "ai2/titan": 32}
BUDGETED = tuple(LIMITS)

_ids = itertools.count()


def job(
    priority=HIGH,
    cm_priority=HIGH,
    slots=8,
    clusters=("ai2/jupiter",),
    assigned_cluster=None,
    is_placed=False,
    queued_at=0.0,
    placed_at=None,
    replica_group_size=0,
    name="job",
    author="jeremym",
):
    """Build a JobView, defaulting to a queued 8-GPU jupiter job."""
    return JobView(
        id=f"job-{next(_ids):04d}",
        name=name,
        author=author,
        priority=priority,
        slots=slots,
        clusters=clusters,
        assigned_cluster=assigned_cluster,
        cm_priority=cm_priority,
        is_placed=is_placed,
        queued_at=queued_at,
        placed_at=placed_at,
        replica_group_size=replica_group_size,
    )


def placed(**kwargs):
    """A job Beaker has put on a node, and so will not raise the priority of."""
    kwargs.setdefault("placed_at", 100.0)
    kwargs.setdefault("assigned_cluster", kwargs.get("clusters", ("ai2/jupiter",))[0])
    kwargs["is_placed"] = True
    return job(**kwargs)


def changes(actions: list[Action]) -> dict[str, int]:
    return {action.job.id: action.to_priority for action in actions}


# --- an oracle written independently of the production accounting ------------


def charged_clusters(job_view: JobView) -> tuple[str, ...]:
    """Which budgets a job's urgent slots land on, derived from first principles.

    Deliberately does not call ``JobView.budget_clusters``: an oracle that reuses
    the code under test cannot catch a bug in that code.
    """
    if job_view.is_placed:
        if job_view.assigned_cluster is None:
            return BUDGETED  # landed somewhere we could not identify
        if job_view.assigned_cluster in LIMITS:
            return (job_view.assigned_cluster,)
        return ()  # landed on a cluster we have no allocation on
    eligible = tuple(c for c in job_view.clusters if c in LIMITS)
    if not job_view.clusters:
        return BUDGETED  # unconstrained: could land anywhere
    return eligible


def usage(jobs, limits=LIMITS) -> dict[str, int]:
    used = dict.fromkeys(limits, 0)
    for job_view in jobs:
        if job_view.priority == URGENT:
            for cluster in charged_clusters(job_view):
                used[cluster] += job_view.slots
    return used


def unmanaged_charge(jobs, limits=LIMITS) -> dict[str, int]:
    """Urgent slots held by jobs the balancer cannot modify."""
    return usage([j for j in jobs if j.managed_cluster(limits) is None], limits)


def apply(jobs: list[JobView], actions: list[Action]) -> list[JobView]:
    """Return the job list as it would be after the actions are applied."""
    new_priority = {action.job.id: action.to_priority for action in actions}
    return [
        replace(j, priority=new_priority[j.id]) if j.id in new_priority else j
        for j in jobs
    ]


# --- resting priority -------------------------------------------------------


def test_resting_priority_drops_urgent_to_high():
    assert resting_priority(URGENT) == HIGH
    assert resting_priority(HIGH) == HIGH
    assert resting_priority(LOW) == LOW


# --- promotion --------------------------------------------------------------


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
    mislabelled = job(priority=HIGH, cm_priority=LOW, slots=8)
    hog = job(priority=URGENT, cm_priority=None, slots=72)
    assert changes(decide([hog, mislabelled], LIMITS)) == {mislabelled.id: LOW}


@pytest.mark.parametrize("cm_priority", [LOW, NORMAL, HIGH, URGENT])
def test_any_labelled_job_can_be_granted_an_urgent_slot(cm_priority):
    queued = job(cm_priority=cm_priority)
    assert changes(decide([queued], LIMITS)) == {queued.id: URGENT}


# --- demotion ---------------------------------------------------------------


def test_over_allocation_demotes_lowest_cm_priority_first():
    keep = placed(priority=URGENT, cm_priority=HIGH, slots=64)
    drop = placed(priority=URGENT, cm_priority=LOW, slots=8)
    result = changes(decide([keep, drop], {"ai2/jupiter": 64}))
    assert result == {drop.id: LOW}


def test_queued_jobs_are_demoted_before_placed_ones_at_the_same_level():
    still_queued = job(priority=URGENT, cm_priority=HIGH, slots=8)
    on_gpu = placed(priority=URGENT, cm_priority=HIGH, slots=8)
    result = changes(decide([still_queued, on_gpu], {"ai2/jupiter": 8}))
    assert result == {still_queued.id: HIGH}


def test_longest_placed_job_is_demoted_first():
    old = placed(priority=URGENT, cm_priority=HIGH, slots=8, placed_at=100.0)
    new = placed(priority=URGENT, cm_priority=HIGH, slots=8, placed_at=900.0)
    result = changes(decide([old, new], {"ai2/jupiter": 8}))
    assert result == {old.id: HIGH}


def test_longest_queued_job_is_promoted_first():
    old = job(cm_priority=HIGH, slots=8, queued_at=100.0)
    new = job(cm_priority=HIGH, slots=8, queued_at=900.0)
    result = changes(decide([old, new], {"ai2/jupiter": 8}))
    assert result == {old.id: URGENT}


def test_demoted_job_returns_to_its_own_label_not_to_high():
    over = placed(priority=URGENT, cm_priority=LOW, slots=8)
    assert changes(decide([over], {"ai2/jupiter": 0})) == {over.id: LOW}


def test_demotions_are_ordered_before_promotions():
    # A partially applied pass must never sit over allocation, so the demotion
    # paying for a promotion has to be attempted first.
    holder = placed(priority=URGENT, cm_priority=NORMAL, slots=8)
    contender = job(cm_priority=HIGH, slots=8)
    actions = decide([contender, holder], {"ai2/jupiter": 8})
    assert [a.is_demotion for a in actions] == [True, False]


# --- placement and promotability --------------------------------------------


def test_placed_job_is_never_promoted():
    # Beaker rejects raising the priority of a job it has already placed.
    on_gpu = placed(priority=HIGH, cm_priority=URGENT)
    assert decide([on_gpu], LIMITS) == []


def test_placed_job_below_its_resting_priority_is_left_alone():
    on_gpu = placed(priority=NORMAL, cm_priority=HIGH)
    assert decide([on_gpu], LIMITS) == []


def test_scheduled_but_not_started_job_is_not_treated_as_queued():
    # A job Beaker has scheduled is initializing: it already holds its slots and
    # Beaker already refuses to raise it, so promoting it would both double-count
    # the slots and emit a call that fails on every pass.
    initializing = job(
        priority=HIGH,
        cm_priority=URGENT,
        is_placed=True,
        assigned_cluster="ai2/jupiter",
        placed_at=500.0,
    )
    assert initializing.is_queued is False
    assert decide([initializing], LIMITS) == []


def test_placed_non_urgent_job_does_not_displace_others():
    blocked = placed(priority=HIGH, cm_priority=URGENT, slots=8)
    queued = job(cm_priority=NORMAL, slots=8)
    result = changes(decide([blocked, queued], {"ai2/jupiter": 8}))
    assert result == {queued.id: URGENT}


# --- accounting -------------------------------------------------------------


def test_multi_cluster_jobs_are_never_modified():
    both = job(clusters=("ai2/titan", "ai2/jupiter"), cm_priority=URGENT)
    assert decide([both], LIMITS) == []


def test_queued_multi_cluster_urgent_job_counts_against_every_cluster():
    both = job(priority=URGENT, cm_priority=None, clusters=("ai2/titan", "ai2/jupiter"))
    on_jupiter = job(cm_priority=HIGH, slots=72, clusters=("ai2/jupiter",))
    on_titan = job(cm_priority=HIGH, slots=32, clusters=("ai2/titan",))
    assert decide([both, on_jupiter, on_titan], LIMITS) == []


def test_queued_job_with_no_cluster_constraint_is_charged_everywhere():
    # It could land anywhere, so charging it to nothing would let the balancer
    # hand out slots that are already spoken for.
    ghost = job(priority=URGENT, cm_priority=None, clusters=(), slots=72)
    assert ghost.budget_clusters(LIMITS) == BUDGETED
    on_jupiter = job(cm_priority=HIGH, slots=8, clusters=("ai2/jupiter",))
    assert decide([ghost, on_jupiter], LIMITS) == []


def test_placed_job_with_unresolvable_cluster_is_charged_everywhere():
    unknown = job(
        priority=URGENT,
        cm_priority=None,
        is_placed=True,
        assigned_cluster=None,
        slots=72,
    )
    assert unknown.budget_clusters(LIMITS) == BUDGETED


def test_placed_job_with_unresolvable_cluster_is_not_managed():
    # It is charged to every budget as a precaution, so managing it would let
    # the walk spend slots on one cluster while the accounting spent them on
    # all of them -- and the pass would end over allocation.
    unknown = job(
        cm_priority=HIGH,
        clusters=("ai2/titan",),
        is_placed=True,
        assigned_cluster=None,
        placed_at=100.0,
        slots=8,
    )
    assert unknown.managed_cluster(LIMITS) is None
    assert decide([unknown], LIMITS) == []


def test_placed_job_on_its_pinned_cluster_is_managed():
    on_titan = job(
        cm_priority=HIGH,
        clusters=("ai2/titan",),
        is_placed=True,
        assigned_cluster="ai2/titan",
        placed_at=100.0,
        slots=8,
    )
    assert on_titan.managed_cluster(LIMITS) == "ai2/titan"


def test_assigned_multi_cluster_job_counts_only_where_it_landed():
    landed = placed(
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
    assert decide([unlabelled, labelled], LIMITS) == []


def test_cluster_allocations_are_independent():
    jupiter = job(clusters=("ai2/jupiter",), cm_priority=HIGH, slots=72)
    titan = job(clusters=("ai2/titan",), cm_priority=HIGH, slots=32)
    result = changes(decide([jupiter, titan], LIMITS))
    assert result == {jupiter.id: URGENT, titan.id: URGENT}


def test_jobs_on_unbudgeted_clusters_are_left_alone():
    saturn = job(clusters=("ai2/saturn",), cm_priority=URGENT, priority=URGENT)
    assert decide([saturn], LIMITS) == []


def test_replica_group_members_are_not_managed():
    # Granting urgent to some ranks and not others wastes the slots granted,
    # and it is unconfirmed whether the priority RPC is per-job or per-source.
    ranks = [job(cm_priority=HIGH, slots=8, replica_group_size=4) for _ in range(4)]
    assert decide(ranks, LIMITS) == []


# --- fitting ----------------------------------------------------------------


def test_first_fit_skips_a_job_that_does_not_fit():
    big = job(cm_priority=HIGH, slots=16, queued_at=0.0)
    small = job(cm_priority=HIGH, slots=8, queued_at=10.0)
    result = changes(decide([big, small], {"ai2/jupiter": 8}))
    assert result == {small.id: URGENT}


def test_a_blocked_level_does_not_reserve_slots_for_itself():
    big = job(cm_priority=HIGH, slots=16)
    small = job(cm_priority=NORMAL, slots=8)
    result = changes(decide([big, small], {"ai2/jupiter": 8}))
    assert result == {small.id: URGENT}


def test_higher_level_job_displaces_lower_level_urgent_jobs():
    holders = [placed(priority=URGENT, cm_priority=NORMAL, slots=4) for _ in range(2)]
    contender = job(cm_priority=HIGH, slots=8)
    result = changes(decide([*holders, contender], {"ai2/jupiter": 8}))
    assert result == {
        holders[0].id: NORMAL,
        holders[1].id: NORMAL,
        contender.id: URGENT,
    }


# --- determinism ------------------------------------------------------------


def test_tied_jobs_are_broken_deterministically_by_id():
    # Whole-second clocks mean a launch wave ties; without an explicit
    # tie-break the outcome would depend on Beaker's listing order.
    def build():
        return [job(cm_priority=HIGH, slots=8, queued_at=500.0) for _ in range(4)]

    jobs = build()
    forward = changes(decide(jobs, {"ai2/jupiter": 16}))
    backward = changes(decide(list(reversed(jobs)), {"ai2/jupiter": 16}))
    assert forward == backward
    assert len(forward) == 2


# --- invariants over randomised populations ---------------------------------


def random_population(rng: random.Random) -> list[JobView]:
    jobs = []
    for _ in range(rng.randint(0, 40)):
        is_placed = rng.random() < 0.5
        clusters = rng.choice(
            [
                ("ai2/jupiter",),
                ("ai2/titan",),
                ("ai2/titan", "ai2/jupiter"),
                ("ai2/saturn",),
                (),  # unconstrained: could land anywhere
            ]
        )
        # A placed job may be running, or scheduled and still initializing.
        assigned = None
        if is_placed and clusters and rng.random() < 0.9:
            assigned = rng.choice(clusters)
        jobs.append(
            job(
                priority=rng.choice([LOW, NORMAL, HIGH, URGENT]),
                cm_priority=rng.choice([None, LOW, NORMAL, HIGH, URGENT]),
                slots=rng.choice([1, 2, 4, 8, 16, 40]),
                clusters=clusters,
                is_placed=is_placed,
                assigned_cluster=assigned,
                placed_at=float(rng.randint(1, 10_000)) if is_placed else None,
                queued_at=float(rng.randint(1, 100)),  # ties are common
                replica_group_size=rng.choice([0, 0, 0, 4]),
            )
        )
    return jobs


@pytest.mark.parametrize("seed", range(300))
def test_allocation_is_never_exceeded_by_our_own_doing(seed):
    """The core guarantee: a pass never pushes usage above the allocation.

    Usage may legitimately start above a limit because of jobs the balancer
    cannot modify. The claim is that it never ends above the limit *or* above
    that unmodifiable charge, whichever is higher.
    """
    rng = random.Random(seed)
    jobs = random_population(rng)
    floor = unmanaged_charge(jobs)
    after = usage(apply(jobs, decide(jobs, LIMITS)))
    for cluster, limit in LIMITS.items():
        assert after[cluster] <= max(
            limit, floor[cluster]
        ), f"{cluster}: -> {after[cluster]}, limit {limit}, floor {floor[cluster]}"


@pytest.mark.parametrize("seed", range(300))
def test_a_pass_always_converges(seed):
    """A second pass must be a no-op, or the cron would flap jobs."""
    rng = random.Random(seed)
    jobs = random_population(rng)
    settled = apply(jobs, decide(jobs, LIMITS))
    assert decide(settled, LIMITS) == []


@pytest.mark.parametrize("seed", range(300))
def test_converges_even_when_every_action_fails(seed):
    """A pass whose calls all fail must not change what the next pass decides.

    The permission fail-soft path means actions can silently not happen; the
    balancer must simply retry the same thing rather than escalate.
    """
    rng = random.Random(seed)
    jobs = random_population(rng)
    first = decide(jobs, LIMITS)
    assert decide(jobs, LIMITS) == first


@pytest.mark.parametrize("seed", range(300))
def test_any_prefix_of_a_pass_stays_within_allocation(seed):
    """Actions are applied one at a time and any of them can fail.

    Every prefix must therefore be safe, which is what ordering demotions first
    buys us.
    """
    rng = random.Random(seed)
    jobs = random_population(rng)
    actions = decide(jobs, LIMITS)
    floor = unmanaged_charge(jobs)
    start = usage(jobs)
    for i in range(len(actions) + 1):
        after = usage(apply(jobs, actions[:i]))
        for cluster, limit in LIMITS.items():
            assert after[cluster] <= max(limit, floor[cluster], start[cluster])


@pytest.mark.parametrize("seed", range(300))
def test_only_labelled_manageable_jobs_are_ever_modified(seed):
    rng = random.Random(seed)
    jobs = random_population(rng)
    for action in decide(jobs, LIMITS):
        assert action.job.cm_priority is not None
        assert action.job.managed_cluster(LIMITS) is not None
        assert action.job.replica_group_size <= 1
        # Beaker refuses to raise a placed job; we must never try.
        if action.job.is_placed:
            assert action.to_priority < action.from_priority


@pytest.mark.parametrize("seed", range(300))
def test_a_job_is_never_left_above_urgent_or_below_its_label(seed):
    rng = random.Random(seed)
    jobs = random_population(rng)
    for action in decide(jobs, LIMITS):
        cm_priority = action.job.cm_priority
        assert cm_priority is not None
        assert action.to_priority in (URGENT, resting_priority(cm_priority))


def test_random_population_reaches_the_states_that_matter():
    """Guard against the property tests quietly becoming vacuous."""
    seen = {
        "unconstrained": 0,
        "initializing": 0,
        "replica": 0,
        "demotion": 0,
        "promotion": 0,
        "over_before": 0,
    }
    for seed in range(300):
        jobs = random_population(random.Random(seed))
        seen["unconstrained"] += sum(1 for j in jobs if not j.clusters)
        seen["initializing"] += sum(
            1 for j in jobs if j.is_placed and j.assigned_cluster is None
        )
        seen["replica"] += sum(1 for j in jobs if j.replica_group_size > 1)
        actions = decide(jobs, LIMITS)
        seen["demotion"] += sum(1 for a in actions if a.is_demotion)
        seen["promotion"] += sum(1 for a in actions if not a.is_demotion)
        before = usage(jobs)
        seen["over_before"] += any(before[c] > LIMITS[c] for c in LIMITS)
    for state, count in seen.items():
        assert count > 10, f"{state} only reached {count} times"
