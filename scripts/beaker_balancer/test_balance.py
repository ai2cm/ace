"""Tests for the balancer's decision logic.

These exercise ``decide`` against constructed job lists, so no Beaker calls are
made. The Beaker-facing layer is covered in ``test_beaker_io.py``.
"""

import copy
import itertools
import random
from dataclasses import replace

import pytest

pytest.importorskip("beaker", reason="beaker-py is not a core fme dependency")

from balance import (  # noqa: E402
    HIGH,
    IMMEDIATE,
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
    replica_group_id="",
    is_session=False,
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
        replica_group_id=replica_group_id,
        is_session=is_session,
    )


def group(count=4, group_id="rg-1", **kwargs):
    """Build the ranks of one replica group, all sharing a spec."""
    return [
        job(replica_group_size=count, replica_group_id=group_id, **kwargs)
        for _ in range(count)
    ]


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
        # immediate outranks urgent, so it occupies a slot just the same.
        if job_view.priority in (URGENT, IMMEDIATE):
            for cluster in charged_clusters(job_view):
                used[cluster] += job_view.slots
    return used


def modifiable(jobs, limits=LIMITS) -> set[str]:
    """Ids of jobs the balancer may change, restated from the rules.

    A rank of a replica group is modifiable only as part of a whole group that
    is uniformly manageable and entirely present, since the group is granted
    urgent all-or-nothing.
    """
    groups: dict[str, list[JobView]] = {}
    for j in jobs:
        key = j.replica_group_id if j.replica_group_size > 1 else j.id
        groups.setdefault(key or j.id, []).append(j)
    allowed: set[str] = set()
    for members in groups.values():
        if any(m.managed_cluster(limits) is None for m in members):
            continue
        if len({m.cm_priority for m in members}) != 1:
            continue
        if len({m.clusters[0] for m in members}) != 1:
            continue
        if len(members) < max(m.replica_group_size for m in members):
            continue
        allowed.update(m.id for m in members)
    return allowed


def unmanaged_charge(jobs, limits=LIMITS) -> dict[str, int]:
    """Allocated slots held by jobs the balancer cannot modify."""
    allowed = modifiable(jobs, limits)
    return usage([j for j in jobs if j.id not in allowed], limits)


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


# --- replica groups ---------------------------------------------------------


def test_replica_group_ranks_are_granted_together():
    ranks = group(4, cm_priority=HIGH, slots=8)
    assert changes(decide(ranks, LIMITS)) == {rank.id: URGENT for rank in ranks}


def test_a_replica_group_is_granted_all_or_nothing():
    # 5 x 16 = 80 against 72: four ranks would fit, but a group cannot be split
    # to fit, so every rank gives up urgent rather than stranding 64 slots on a
    # job that still cannot start.
    ranks = group(5, priority=URGENT, cm_priority=HIGH, slots=16)
    assert changes(decide(ranks, LIMITS)) == {rank.id: HIGH for rank in ranks}


def test_a_replica_group_is_charged_as_the_sum_of_its_ranks():
    big = group(4, group_id="big", cm_priority=HIGH, slots=16)
    small = job(cm_priority=NORMAL, slots=16)
    # The 64-slot group wins its level first, leaving 8 — too few for the
    # 16-slot normal job behind it.
    result = changes(decide([*big, small], LIMITS))
    assert result == {**{rank.id: URGENT for rank in big}, small.id: NORMAL}


def test_a_replica_group_with_an_unlabelled_rank_is_left_alone():
    ranks = group(3, cm_priority=HIGH)
    ranks.append(job(cm_priority=None, replica_group_size=4, replica_group_id="rg-1"))
    assert decide(ranks, LIMITS) == []


def test_a_replica_group_whose_ranks_disagree_on_their_label_is_left_alone():
    ranks = group(3, cm_priority=HIGH)
    ranks.append(job(cm_priority=LOW, replica_group_size=4, replica_group_id="rg-1"))
    assert decide(ranks, LIMITS) == []


def test_a_partially_visible_replica_group_is_left_alone():
    # fetch_jobs reads only unfinished jobs in one workspace, so a group can
    # come back short. Granting the visible ranks would half-grant the real one.
    assert decide(group(4, group_id="rg-1")[:3], LIMITS) == []


def test_a_replica_group_with_a_placed_rank_below_urgent_is_not_promoted():
    # Beaker cannot raise the placed rank, so granting urgent would split the
    # group. It is excluded from the walk instead.
    ranks = [
        placed(
            priority=HIGH,
            cm_priority=URGENT,
            replica_group_size=2,
            replica_group_id="rg-1",
            slots=8,
        ),
        job(
            priority=HIGH,
            cm_priority=URGENT,
            replica_group_size=2,
            replica_group_id="rg-1",
            slots=8,
        ),
    ]
    contender = job(cm_priority=NORMAL, slots=8)
    result = changes(decide([*ranks, contender], {"ai2/jupiter": 8}))
    assert result == {contender.id: URGENT}


# --- sessions and immediate priority ----------------------------------------


def test_an_interactive_session_is_never_modified():
    session = job(priority=URGENT, cm_priority=LOW, is_session=True, slots=8)
    assert decide([session], {"ai2/jupiter": 0}) == []


def test_an_interactive_session_still_consumes_the_allocation():
    session = job(priority=URGENT, cm_priority=None, is_session=True, slots=72)
    contender = job(priority=URGENT, cm_priority=HIGH, slots=8)
    assert changes(decide([session, contender], LIMITS)) == {contender.id: HIGH}


def test_an_immediate_job_is_never_demoted():
    urgent_now = job(priority=IMMEDIATE, cm_priority=LOW, slots=8)
    assert decide([urgent_now], {"ai2/jupiter": 0}) == []


def test_an_immediate_job_consumes_the_allocation():
    urgent_now = job(priority=IMMEDIATE, cm_priority=None, slots=72)
    contender = job(priority=URGENT, cm_priority=HIGH, slots=8)
    assert changes(decide([urgent_now, contender], LIMITS)) == {contender.id: HIGH}


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
    """Build a random workspace of lone jobs, replica groups and sessions.

    Ranks of a group share a spec but vary in placement, and a group is
    occasionally emitted short a rank or with one rank's label perturbed, so
    the uniformly-manageable check is genuinely exercised.
    """
    jobs = []
    for index in range(rng.randint(0, 25)):
        clusters = rng.choice(
            [
                ("ai2/jupiter",),
                ("ai2/titan",),
                ("ai2/titan", "ai2/jupiter"),
                ("ai2/saturn",),
                (),  # unconstrained: could land anywhere
            ]
        )
        size = rng.choice([0, 0, 0, 4])
        shared = {
            "cm_priority": rng.choice([None, LOW, NORMAL, HIGH, URGENT]),
            "slots": rng.choice([1, 2, 4, 8, 16, 40]),
            "clusters": clusters,
            "replica_group_size": size,
            "replica_group_id": f"rg-{index}" if size else "",
            "is_session": rng.random() < 0.1,
        }
        priority = rng.choice([LOW, NORMAL, HIGH, URGENT, IMMEDIATE])
        ranks = max(1, size)
        if size and rng.random() < 0.2:
            ranks -= 1  # a group we can only partly see
        for rank in range(ranks):
            is_placed = rng.random() < 0.5
            # A placed job may be running, or scheduled and still initializing.
            assigned = None
            if is_placed and clusters and rng.random() < 0.9:
                assigned = rng.choice(clusters)
            overrides = {}
            if size and rank == 0 and rng.random() < 0.15:
                overrides["cm_priority"] = rng.choice([None, LOW, URGENT])
            jobs.append(
                job(
                    **{**shared, **overrides},
                    priority=(
                        rng.choice([LOW, NORMAL, HIGH, URGENT, IMMEDIATE])
                        if rng.random() < 0.15
                        else priority
                    ),
                    is_placed=is_placed,
                    assigned_cluster=assigned,
                    placed_at=float(rng.randint(1, 10_000)) if is_placed else None,
                    queued_at=float(rng.randint(1, 100)),  # ties are common
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
def test_decide_does_not_mutate_the_jobs_it_is_given(seed):
    """decide reads the world; run_pass is what changes it.

    This is what makes a pass whose calls all fail decide the same thing again
    rather than escalate. The failure behaviour itself is tested against
    run_pass, where the applying actually happens.
    """
    rng = random.Random(seed)
    jobs = random_population(rng)
    before = copy.deepcopy(jobs)
    decide(jobs, LIMITS)
    assert jobs == before


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
        assert action.job.id in modifiable(jobs)
        assert not action.job.is_session
        assert action.from_priority != IMMEDIATE
        assert action.to_priority != IMMEDIATE
        # Beaker refuses to raise a placed job; we must never try.
        if action.job.is_placed:
            assert action.to_priority < action.from_priority


def half_urgent(members) -> bool:
    holders = sum(1 for member in members if member.priority == URGENT)
    return 0 < holders < len(members)


@pytest.mark.parametrize("seed", range(300))
def test_a_replica_group_never_half_holds_urgent(seed):
    """The guarantee group granularity exists for.

    A group holding urgent on only some ranks cannot start until every rank is
    placed, so it waits at the priority of its lowest rank while the granted
    ranks hold slots the allocation has already spent.

    Splits *below* urgent are left alone deliberately: they cost the allocation
    nothing, and raising a queued rank whose siblings are stuck at a lower
    priority is what gets the group placed.
    """
    rng = random.Random(seed)
    jobs = random_population(rng)
    after = {j.id: j for j in apply(jobs, decide(jobs, LIMITS))}
    groups: dict[str, list[JobView]] = {}
    for j in jobs:
        if j.replica_group_size > 1 and j.replica_group_id:
            groups.setdefault(j.replica_group_id, []).append(j)
    for group_id, members in groups.items():
        if half_urgent(members):
            continue  # the balancer did not create this, and may not own it
        settled = [after[m.id] for m in members]
        assert not half_urgent(settled), f"{group_id} was left half-urgent"


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
        "replica_managed": 0,
        "session": 0,
        "immediate": 0,
        "demotion": 0,
        "promotion": 0,
        "over_before": 0,
    }
    for seed in range(300):
        jobs = random_population(random.Random(seed))
        allowed = modifiable(jobs)
        seen["unconstrained"] += sum(1 for j in jobs if not j.clusters)
        seen["initializing"] += sum(
            1 for j in jobs if j.is_placed and j.assigned_cluster is None
        )
        seen["replica"] += sum(1 for j in jobs if j.replica_group_size > 1)
        seen["replica_managed"] += sum(
            1 for j in jobs if j.replica_group_size > 1 and j.id in allowed
        )
        seen["session"] += sum(1 for j in jobs if j.is_session)
        seen["immediate"] += sum(1 for j in jobs if j.priority == IMMEDIATE)
        actions = decide(jobs, LIMITS)
        seen["demotion"] += sum(1 for a in actions if a.is_demotion)
        seen["promotion"] += sum(1 for a in actions if not a.is_demotion)
        before = usage(jobs)
        seen["over_before"] += any(before[c] > LIMITS[c] for c in LIMITS)
    for state, count in seen.items():
        assert count > 10, f"{state} only reached {count} times"
