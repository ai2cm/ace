"""Tier 2: the agreement against a real communicator, several ranks, no signals.

These tests send no signals. They drive the pending state through
`PendingStop.request`, which is public precisely so that the collective is
testable without sending a signal into a live pytest session -- a signal there
would tear the session's own backend down.

**Every test builds a throwaway agreement group and never releases it.** Both
halves of that are mandatory and neither is tidiness.

*Throwaway*, because a gloo context that has timed out is in an undefined state,
so the session's process-wide group must never be the one that fails -- every
later test in the session would inherit the damage. That is why
`build_stop_agreement` exists beside `new_stop_agreement`: the session's default
group never changes, so the cache in `new_stop_agreement` would hand every test
the session's own group, which is exactly the one that must not fail.

*Never released*, because ``destroy_process_group(group)`` reclaims nothing --
the group *object* dying is what runs ``~ProcessGroupGloo()``, and after a
timed-out exchange that destructor's wait never ends. CPython's refcounting makes
an ordinary function return enough to trigger it, so a test that merely returns
is the hazard; no ``gc.collect()`` is required. **The autouse SIGALRM cannot
rescue it**: a Python signal handler does not run while the main thread is inside
a C++ condition-variable wait, so the session would hang until CI killed it.
Hence `_leak_forever`.
"""

import contextlib
import time
import weakref
from collections.abc import Iterator

import pytest
import torch
import torch.distributed

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.cooperative_stop import (
    _MIN_DEADLINE,
    CooperativeStop,
    PeerStopRequested,
    UnequalLoopLength,
    _cooperative_scope,
)
from fme.core.distributed.shutdown import PendingStop, StopReason
from fme.core.distributed.stop_agreement import (
    GlooStopAgreement,
    build_stop_agreement,
    is_deadline_expiry,
)

# Groups this module has finished with and must never let go of. See the module
# docstring: releasing one after a timed-out exchange hangs the pytest session.
_leak_forever: list[GlooStopAgreement] = []

# Long enough that an exchange every rank really reaches returns inside it, short
# enough that a mistake becomes a raise rather than a hang the alarm cannot
# interrupt. Not `no_local_event_deadline()`, which is the default group's own
# timeout -- tens of minutes.
_VERIFY_TIMEOUT = 30.0

_SKIP_SINGLE_RANK = "the agreement is only meaningful with more than one rank"


def _marker_lines(event: str, captured: str) -> list[str]:
    prefix = f"fme-stop:{event} "
    return [line for line in captured.splitlines() if line.startswith(prefix)]


def _require_ranks() -> Distributed:
    dist = Distributed.get_instance()
    if dist.world_size == 1:
        pytest.skip(_SKIP_SINGLE_RANK)
    return dist


@contextlib.contextmanager
def _throwaway_agreement(destroy: bool = True) -> Iterator[GlooStopAgreement]:
    """A group of this test's own, leaked deliberately on the way out.

    Args:
        destroy: Clear torch's bookkeeping for the group here. ``False`` for a
            test that destroys it itself, because that is the thing under test and
            destroying it twice asks torch to forget a group it has forgotten.
    """
    dist = Distributed.get_instance()
    agreement = build_stop_agreement(dist.world_size)
    # before anything that could fail, so no path can be the last reference
    _leak_forever.append(agreement)
    try:
        yield agreement
    finally:
        if destroy:
            # returns at once and reclaims nothing, which is the point
            torch.distributed.destroy_process_group(agreement.group)
        # every rank leaves this test together, so the next one starts clean
        dist.barrier()


@pytest.mark.parallel
def test_every_rank_leaves_the_loop_at_the_same_index():
    """The whole point, with a real communicator.

    One rank is told to stop at iteration 5. Every rank must leave having
    completed exactly the same batches -- which no flag-only design can deliver,
    because a rank standing at a boundary with its own flag set cannot know
    whether a peer has already entered the next iteration, and a peer that has is
    stranded in a collective nobody will complete.
    """
    dist = _require_ranks()
    notice_at = 5
    with _throwaway_agreement() as agreement:
        pending = PendingStop(budget=10.0)
        with _cooperative_scope(pending, agreement) as stop:
            completed = 0
            for index in range(20):
                if dist.rank == dist.world_size - 1 and index == notice_at:
                    pending.request(15)  # SIGTERM, without sending one
                completed += 1
                if stop.agreed(index):
                    break

        assert completed == notice_at + 1
        # and the ranks agree on that, rather than each being locally consistent
        _, high, low = agreement.exchange(0, completed, _VERIFY_TIMEOUT)
        assert high == low == notice_at + 1
        # a rank that was never signalled still knows it is leaving
        if dist.rank != dist.world_size - 1:
            assert pending.requested is False
            assert pending.peer_stop is True


@pytest.mark.parallel
@pytest.mark.medium_duration
def test_agreement_is_bounded_when_a_peer_never_joins(capfd):
    """A rank that is leaving does not wait for a peer indefinitely.

    This is the case the mechanism exists for: a peer whose main thread is inside
    a C-level call never reaches a boundary, so the ranks that did reach one give
    up on it and tear down without it. The bound is the per-operation deadline,
    which is the design's only bound -- a gloo group can neither be created short
    nor shortened afterwards.
    """
    dist = _require_ranks()
    absent = dist.world_size - 1
    # Above `_MIN_DEADLINE`, so that the *budget* is what bounds the wait, which is
    # what this test is about. A budget under the floor is floored up to it, and
    # would also make `PendingStop`'s own overrun warning -- armed at twice the
    # budget -- fire during a wait that is going to end normally.
    budget = _MIN_DEADLINE + 1.0
    with _throwaway_agreement() as agreement:
        if dist.rank == absent:
            return  # never joins, as a wedged rank never does

        # `work.wait` must *raise* on expiry rather than return `False`. The whole
        # bound rests on that, and it is a private torch behaviour rather than a
        # documented one, so it is asserted rather than assumed. Passed straight to
        # `exchange`, so no floor applies and the probe stays short.
        with pytest.raises(RuntimeError) as caught:
            agreement.exchange(0, 0, 1.0)
        assert is_deadline_expiry(caught.value), caught.value

        pending = PendingStop(budget=budget)
        pending.request(15)
        started = time.monotonic()
        with _cooperative_scope(pending, agreement) as stop:
            assert stop.agreed(1) is True
        elapsed = time.monotonic() - started
        pending.close()

    assert elapsed < budget + 2.0, elapsed
    captured = capfd.readouterr().err
    assert len(_marker_lines("agreement-timeout", captured)) == 1
    # the only line that says the group was left behind rather than reclaimed
    assert len(_marker_lines("agreement-abandoned", captured)) == 1


@pytest.mark.parallel
@pytest.mark.medium_duration
def test_agreement_without_a_pending_stop_has_no_deadline_of_its_own(capfd):
    """The false-positive guard, with real collectives.

    A rank with nothing pending of its own has no reason to hurry and no standing
    to declare a peer dead, so it must not hold a short deadline. If it did, root
    walking out of a multi-second periodic checkpoint write -- which happens on a
    schedule, every ~1000 batches -- would make every other rank hard-exit a
    perfectly healthy job, causing the exact fault this mechanism prevents.
    """
    dist = _require_ranks()
    late = 0
    lateness = 2.0
    with _throwaway_agreement() as agreement:
        # configured, but never armed: nothing calls `request`, so
        # `seconds_remaining()` stays `None` and the group's own timeout applies
        pending = PendingStop(budget=0.2)
        with _cooperative_scope(pending, agreement) as stop:
            if dist.rank == late:
                time.sleep(lateness)  # standing in for root's checkpoint write
            assert stop.agreed(0) is False

        assert pending.seconds_remaining() is None
        assert stop.abandoned is False

    assert _marker_lines("agreement-timeout", capfd.readouterr().err) == []


@pytest.mark.parallel
def test_the_agreement_spans_spatial_co_ranks_where_reduce_max_does_not():
    """The agreement must be world-wide, which is why it is not `reduce_max`.

    Under a spatially-parallel backend `reduce_max` reduces over the "data" axis
    of the device mesh, while the teardown destroys every group in the process. So
    a stop originating on a spatial co-rank would never reach its data-parallel
    peers, and reusing `reduce_max` would be a silent correctness bug under two of
    the three configurations CI's GPU job runs.
    """
    dist = _require_ranks()
    if dist.world_size == dist.total_data_parallel_ranks:
        pytest.skip("no spatial co-ranks for the agreement to span")
    originating = dist.world_size - 1
    reason = StopReason.SIGNAL.value if dist.rank == originating else 0

    with _throwaway_agreement() as agreement:
        reduced, _, _ = agreement.exchange(reason, 0, _VERIFY_TIMEOUT)
        assert reduced == StopReason.SIGNAL.value, "every rank must see the stop"

        # rank 0 sits at (h, w) = (0, 0) and the originating rank at
        # (H - 1, W - 1), so with spatial parallelism they are in different data
        # groups and the data-parallel reduction cannot carry the stop
        over_data = torch.tensor([reason], dtype=torch.int64, device=get_device())
        dist.reduce_max(over_data)
        if dist.rank == 0:
            assert int(over_data.item()) == 0


@pytest.mark.parallel
def test_the_agreement_group_is_independent_of_the_gradient_communicator():
    """An exchange is never matched against a collective on the default group.

    NCCL matches collectives by issue order on a communicator rather than by tag,
    so a control collective on the gradient communicator would be safe only while
    every rank issued the identical sequence -- which is exactly the invariant a
    stop breaks. On a group of its own a rank that stops issuing exchanges
    mismatches nothing.

    **Only really meaningful under NCCL** (CI's ``torch/1/1`` GPU configuration).
    gloo tolerates the ordering this drives, so the CPU run is a smoke test that
    the interleaving does not deadlock.
    """
    dist = _require_ranks()
    with _throwaway_agreement() as agreement:
        # issued asynchronously by every rank first, so that the rank-dependent
        # ordering below cannot deadlock: neither group waits on the other
        gradient = torch.tensor(
            [float(dist.rank)], dtype=torch.float32, device=get_device()
        )
        work = torch.distributed.all_reduce(gradient, async_op=True)

        if dist.rank % 2 == 1:
            reduced = agreement.exchange(0, 7, _VERIFY_TIMEOUT)
            work.wait()
        else:
            work.wait()
            reduced = agreement.exchange(0, 7, _VERIFY_TIMEOUT)

        expected = float(sum(range(dist.world_size)))
        assert gradient.item() == pytest.approx(expected)
        assert reduced == (0, 7, 7)


@pytest.mark.parallel
def test_a_peer_exception_between_iterations_stops_every_rank(capfd):
    """A rank raising after the body's last collective does not strand its peers.

    That window is the whole of what the exception rider covers, and it is
    narrower than it looks because the body holds two collectives, not one: the
    gradient all-reduce every iteration, and a metrics reduction every hundredth.
    A raise before either leaves the peers stranded in it by the raise itself,
    which no mechanism at this seam can fix.

    The index the raising rank contributes is the point of the last assertion. It
    never finished the body, so its own counter is one behind its peers' -- and
    contributing that would fire ``index-mismatch`` on exactly the path this test
    covers, turning the design's one self-check into noise.
    """
    dist = _require_ranks()
    raiser = dist.world_size - 1
    raise_at = 3

    class _Boom(RuntimeError):
        pass

    def run() -> None:
        pending = PendingStop(budget=10.0)
        with _cooperative_scope(pending, agreement) as stop:
            for index in range(10):
                # the gradient all-reduce
                every = torch.ones(1, device=get_device())
                torch.distributed.all_reduce(every)
                if index == raise_at:
                    # and the every-hundredth-batch metrics reduction, on the very
                    # iteration the raise happens, so the raise really is after
                    # *both* of the body's collectives
                    gated = torch.ones(1, device=get_device())
                    torch.distributed.all_reduce(gated)
                    if dist.rank == raiser:
                        raise _Boom("something went wrong between iterations")
                if stop.agreed(index):
                    break
        pending.close()

    with _throwaway_agreement() as agreement:
        expected = _Boom if dist.rank == raiser else PeerStopRequested
        with pytest.raises(expected):
            run()

    captured = capfd.readouterr().err
    assert _marker_lines("index-mismatch", captured) == []
    agreed = _marker_lines("stop-agreed", captured)
    assert len(agreed) == 1
    assert f"batch={raise_at}" in agreed[0]
    assert "reason=EXCEPTION" in agreed[0]


@pytest.mark.parallel
@pytest.mark.medium_duration
def test_destroying_an_abandoned_group_returns_without_reclaiming_it(capfd):
    """The bound after a timed-out exchange is exiting, not getting the group back.

    ``destroy_process_group`` returns promptly and reclaims nothing -- which is
    what makes the teardown reachable at all on this path, and hence the restart
    checkpoint. The price is that the group is kept for the life of the process:
    the only thing that drains gloo's work queue is ``~ProcessGroupGloo()``, whose
    wait carries no deadline, so paying the reference back is unbounded.

    This test deliberately does **not** drop the reference. Doing so would hang
    the session, not fail the test.
    """
    dist = _require_ranks()
    absent = dist.world_size - 1
    with _throwaway_agreement(destroy=False) as agreement:
        if dist.rank == absent:
            torch.distributed.destroy_process_group(agreement.group)
            return  # never joins, so it has nothing to give up on

        # above `_MIN_DEADLINE`, for the reason given in
        # `test_agreement_is_bounded_when_a_peer_never_joins`
        pending = PendingStop(budget=_MIN_DEADLINE + 1.0)
        pending.request(15)
        with _cooperative_scope(pending, agreement) as stop:
            assert stop.agreed(2) is True
        assert stop.abandoned is True
        pending.close()

        alive = weakref.ref(agreement.group)
        started = time.monotonic()
        torch.distributed.destroy_process_group(agreement.group)
        elapsed = time.monotonic() - started

        assert elapsed < 1.0, elapsed
        assert alive() is not None, "the group was reclaimed, which can hang"
        assert len(_marker_lines("agreement-abandoned", capfd.readouterr().err)) == 1


@pytest.mark.parallel
def test_unequal_loop_lengths_raise_on_every_rank_at_loop_entry():
    """One exchange at loop entry, because the boundary ones could not see this.

    An unequal per-rank batch count holds by construction today on both sampler
    paths, so this is insurance against that construction changing. If it did, the
    short rank would leave the loop and exchange nothing while the long rank sat
    alone in one more exchange with the default group's own timeout on it -- and the
    ±index
    diagnostic would be blind to it, comparing values contributed to an exchange
    only one rank is in.

    Every rank reads the same reduced values, so every rank raises. That is what
    makes this a symmetric precondition failure rather than one rank unilaterally
    terminating a job.
    """
    dist = _require_ranks()
    with _throwaway_agreement() as agreement:
        equal = CooperativeStop(PendingStop(budget=10.0), agreement)
        equal.assert_equal_loop_length(100)  # raises on no rank

        unequal = CooperativeStop(PendingStop(budget=10.0), agreement)
        length = 99 if dist.rank == dist.world_size - 1 else 100
        with pytest.raises(UnequalLoopLength):
            unequal.assert_equal_loop_length(length)


@pytest.mark.parallel
def test_the_agreement_group_carries_the_backends_world_size():
    """The evidence line's ``world=`` has to come from the group, not the caller.

    A reader counts ``stop-agreed`` lines against it to find the rank that never
    reached the boundary, so a wrong value would make the absence unreadable.
    """
    dist = _require_ranks()
    # at more than one rank this also proves the backend joined a real group
    # rather than falling back to `SoloStopAgreement`, whose world size is 1 --
    # which would make every stop unilateral and strand exactly the peers the
    # mechanism exists to keep
    assert dist.stop_agreement().world_size == dist.world_size
    # the same object every call, so nothing issues a second unmatched `new_group`
    assert dist.stop_agreement() is dist.stop_agreement()
