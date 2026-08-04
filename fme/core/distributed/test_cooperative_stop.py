"""Tier 1: the agreement algorithm, single process, no process group at all.

These tests never create a communicator. The agreement is a scripted or
lockstep-threaded stand-in, which is legitimate because what is under test is the
algorithm -- which deadline a rank applies, which index it contributes, what it
does with a reduced value. Every claim that turns on a real collective lives in
`parallel_tests/test_cooperative_stop.py`, and the exit paths live in
`test_cooperative_stop_exit.py`.

They are unmarked, so they run in every tier including ``--very-fast``, and each
takes milliseconds.
"""

import logging
import signal
import threading
import time
import weakref
from typing import cast

import pytest
import torch
import torch.distributed

from fme.core.distributed import stop_agreement as stop_agreement_module
from fme.core.distributed.cooperative_stop import (
    _MIN_DEADLINE,
    CooperativeStop,
    PeerStopRequested,
    _cooperative_scope,
    no_local_event_deadline,
)
from fme.core.distributed.shutdown import (
    PendingStop,
    StopReason,
    defer_termination,
    handle_termination_signals,
)
from fme.core.distributed.stop_agreement import StopAgreement

_BARRIER_TIMEOUT = 5.0

# With no default group to read, the no-local-event deadline falls back to the
# agreement group's own timeout. Read through the function so a test never has to
# name the figure.
_NO_LOCAL_EVENT_DEADLINE = no_local_event_deadline()


def _marker_lines(event: str, captured: str) -> list[str]:
    prefix = f"fme-stop:{event} "
    return [line for line in captured.splitlines() if line.startswith(prefix)]


class _SpyAgreement(StopAgreement):
    """Records every exchange and returns the scripted peers' answer, reduced.

    The reduction over ``reason`` is honoured rather than discarded. A stub that
    returned its scripted value unreduced could answer ``0`` to a rank that had
    just contributed ``1``, which a real ``MAX`` never does -- so every assertion
    made around such a stub would be about an outcome production cannot produce.
    """

    def __init__(
        self, reason: int = 0, high: int | None = None, low: int | None = None
    ):
        self.calls: list[tuple[int, int, float]] = []
        self._reason = reason
        self._high = high
        self._low = low

    @property
    def world_size(self) -> int:
        return 4

    @property
    def abandoned(self) -> bool:
        return False

    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        self.calls.append((reason, index, timeout))
        high = index if self._high is None else self._high
        low = index if self._low is None else self._low
        return max(self._reason, reason), high, low

    @property
    def timeouts(self) -> list[float]:
        return [timeout for _, _, timeout in self.calls]


class _WedgedPeerAgreement(StopAgreement):
    """Every exchange expires, exactly as one against a wedged peer does."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float]] = []
        self._abandoned = False

    @property
    def world_size(self) -> int:
        return 2

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        self.calls.append((reason, index, timeout))
        self._abandoned = True
        raise RuntimeError("Operation timed out!")


class _LockstepAgreement(StopAgreement):
    """A real ``MAX`` over several logical ranks running as threads.

    One thread per logical rank, meeting at a `threading.Condition`, so the
    reduction is genuinely across ranks and genuinely simultaneous -- which is
    what the claim under test is about -- without a launcher or a communicator.
    """

    def __init__(self, world_size: int):
        self._world_size = world_size
        self._round = 0
        self._contributions: list[tuple[int, int, int]] = []
        # kept per round rather than overwritten, so a fast rank starting the next
        # round cannot make a slow one read the wrong answer
        self._reduced: dict[int, tuple[int, int, int]] = {}
        self._met = threading.Condition()

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def abandoned(self) -> bool:
        return False

    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        with self._met:
            this_round = self._round
            self._contributions.append((reason, index, -index))
            if len(self._contributions) == self._world_size:
                contributions = self._contributions
                self._reduced[this_round] = (
                    max(payload[0] for payload in contributions),
                    max(payload[1] for payload in contributions),
                    max(payload[2] for payload in contributions),
                )
                self._contributions = []
                self._round += 1
                self._met.notify_all()
            else:
                give_up_at = time.monotonic() + _BARRIER_TIMEOUT
                while this_round not in self._reduced:
                    if time.monotonic() > give_up_at:
                        raise AssertionError(
                            f"only {len(self._contributions)} of "
                            f"{self._world_size} logical ranks reached round "
                            f"{this_round}"
                        )
                    self._met.wait(0.05)
            reduced = self._reduced[this_round]
        return reduced[0], reduced[1], -reduced[2]


def _stop(
    agreement: StopAgreement, budget: float = 10.0, first_index: int = 0
) -> tuple[CooperativeStop, PendingStop]:
    pending = PendingStop(budget)
    return CooperativeStop(pending, agreement, first_index=first_index), pending


def test_stop_is_agreed_at_the_same_index_on_every_logical_rank():
    """Every rank leaves having completed the same number of batches.

    This is the whole point of the mechanism, and the reason a flag alone will
    not do: ranks notice at different indices -- legitimately, because root
    spends seconds inside a periodic checkpoint write while its peers do not --
    so the noticing indices have to be collapsed to one value by communication
    before any rank commits to entering the next iteration.
    """
    world_size = 4
    notice_at = {0: 3, 2: 5}
    agreement = _LockstepAgreement(world_size)
    completed: dict[int, int] = {}
    reasons: dict[int, StopReason | None] = {}

    def run(rank: int) -> None:
        stop, pending = _stop(agreement)
        count = 0
        for index in range(20):
            if notice_at.get(rank) == index:
                pending.request(signal.SIGTERM)
            count += 1
            if stop.agreed(index):
                break
        completed[rank] = count
        reasons[rank] = StopReason.SIGNAL if pending.requested else None
        pending.close()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_BARRIER_TIMEOUT * 2)

    assert len(completed) == world_size
    # the earliest noticing index is 3, so every rank completes batches 0-3
    assert set(completed.values()) == {4}
    # and only the rank that was signalled recorded a signal; the others joined
    # the very exchange their peer entered
    assert reasons == {0: StopReason.SIGNAL, 1: None, 2: None, 3: None}


def test_a_rank_with_no_signal_applies_the_group_deadline():
    """The false-positive guard, at unit level.

    A short deadline on a healthy rank fires on a healthy job and hard-exits it,
    causing the exact fabric fault this mechanism exists to prevent. So a rank
    with no local event of its own passes the deadline the gradient all-reduce
    already carries, which therefore cannot fire on a job that is not already
    dead.
    """
    agreement = _SpyAgreement()
    stop, pending = _stop(agreement, budget=10.0)

    assert stop.agreed(0) is False
    assert stop.agreed(1) is False
    assert agreement.timeouts == [
        _NO_LOCAL_EVENT_DEADLINE,
        _NO_LOCAL_EVENT_DEADLINE,
    ]

    pending.request(signal.SIGTERM)
    # true, not false: this rank contributed `SIGNAL`, and a real `MAX` returns at
    # least what the caller put in
    assert stop.agreed(2) is True
    pending.close()

    assert agreement.timeouts[2] < _NO_LOCAL_EVENT_DEADLINE
    assert agreement.timeouts[2] == pytest.approx(10.0, abs=0.5)


def test_the_budget_is_absolute_from_the_local_event_not_per_boundary():
    """A rank cannot accumulate a fresh budget at each of several boundaries.

    If it could, a rank passing several boundaries while a peer never arrives
    would wait indefinitely in budget-sized steps, and the bound the design
    claims would not exist.
    """
    agreement = _SpyAgreement()
    stop, pending = _stop(agreement, budget=10.0)
    pending.request(signal.SIGTERM)
    try:
        for index in range(3):
            time.sleep(0.02)
            stop.agreed(index)
    finally:
        pending.close()

    assert agreement.timeouts == sorted(agreement.timeouts, reverse=True)
    assert agreement.timeouts[0] > agreement.timeouts[-1]
    # and every one of them is a remainder of the same 10s, not a fresh 10s
    assert all(timeout < 10.0 for timeout in agreement.timeouts)


def test_an_expired_budget_exchanges_at_the_floor(capfd):
    """A spent budget means this rank was late, not that a peer was missing.

    So it still exchanges -- in the dominant case its peers are already blocked
    in that exchange and it returns in microseconds -- and the deadline is
    floored. The floor is not cosmetic: torch reads a zero-length timeout as *no*
    timeout, and the pybind caster truncates a `timedelta` to integral
    milliseconds, so an unfloored sliver of budget would wait the default group's
    own timeout.
    """
    agreement = _SpyAgreement()
    stop, pending = _stop(agreement, budget=-1.0)  # already expired when armed
    pending.request(signal.SIGTERM)
    try:
        # true, not false: this rank contributed `SIGNAL` and a real `MAX` returns
        # at least that. What the test is about is the deadline, below.
        assert stop.agreed(7) is True
    finally:
        pending.close()

    assert agreement.timeouts == [_MIN_DEADLINE]
    assert len(_marker_lines("agreement-expired", capfd.readouterr().err)) == 1

    # a sub-millisecond remainder takes the same path rather than degenerating
    sliver = _SpyAgreement()
    stop, pending = _stop(sliver, budget=0.0004)
    pending.request(signal.SIGTERM)
    try:
        stop.agreed(8)
    finally:
        pending.close()

    assert sliver.timeouts == [_MIN_DEADLINE]
    # no deadline anywhere is ever zero, on any path
    assert all(timeout > 0.0 for timeout in agreement.timeouts + sliver.timeouts)


def test_an_index_mismatch_is_reported_and_does_not_force_a_stop(capfd):
    """The symmetry check is diagnostic only, deliberately.

    A desynchronised run is going to hang anyway, and a new mechanism
    unilaterally terminating healthy-looking jobs on a self-diagnosis is a worse
    failure mode than the one it reports.
    """
    agreement = _SpyAgreement(reason=0, high=12, low=11)
    stop, pending = _stop(agreement)

    assert stop.agreed(11) is False

    captured = capfd.readouterr().err
    mismatch = _marker_lines("index-mismatch", captured)
    assert len(mismatch) == 1
    assert "min=11" in mismatch[0] and "max=12" in mismatch[0]
    assert _marker_lines("stop-agreed", captured) == []
    pending.close()


def _world(name: str) -> torch.distributed.ProcessGroup:
    """One default group, standing in only for its identity.

    A `cast` rather than a real group: what `new_stop_agreement` does with this is
    compare it by identity and take a weak reference to it, so any object with a
    ``__weakref__`` presents the state a call site presents.
    """
    return cast(torch.distributed.ProcessGroup, type(name, (), {})())


def test_a_superseded_agreement_group_is_leaked_not_released(monkeypatch):
    """Reclaiming an agreement group is the one thing the design must never do.

    Dropping the last reference to a gloo group runs ``~ProcessGroupGloo()``,
    which waits without bound for a work queue an abandoned exchange leaves
    non-empty -- and nothing else drains it, gloo overriding neither
    ``shutdown()`` nor ``abort()``. So a superseded group is *moved* to a
    process-lifetime list, never replaced with ``None``.

    The state is the one a real call site presents: two successive default groups,
    which is what a ``destroy_process_group()`` followed by a fresh
    ``init_process_group()`` in one process leaves behind. It is not
    ``is_initialized()`` going false -- every production caller runs immediately
    after ``init_process_group``, so it never sees that.
    """
    built: list[_SpyAgreement] = []

    def build(world_size: int) -> _SpyAgreement:
        agreement = _SpyAgreement()
        built.append(agreement)
        return agreement

    monkeypatch.setattr(stop_agreement_module, "build_stop_agreement", build)
    monkeypatch.setattr(stop_agreement_module, "_agreement", None)
    monkeypatch.setattr(stop_agreement_module, "_agreement_world", None)
    monkeypatch.setattr(stop_agreement_module, "_leaked", [])

    before, after = _world("before"), _world("after")
    first = stop_agreement_module.new_stop_agreement(4, before)

    # the same world asks for the same group, rather than a second `new_group`
    # some ranks would not match
    assert stop_agreement_module.new_stop_agreement(4, before) is first
    assert len(built) == 1

    alive = weakref.ref(first)
    second = stop_agreement_module.new_stop_agreement(4, after)

    assert second is not first
    assert stop_agreement_module._leaked == [first]
    del first
    assert alive() is not None, "the superseded group was released, which can hang"
    assert len(built) == 2


def test_a_collected_world_invalidates_the_cached_agreement_group(monkeypatch):
    """The cached world is held weakly, so a world that has gone invalidates.

    A strong reference would make `stop_agreement` a potential last holder of the
    *default* group, whose destructor carries the same unbounded wait as the
    agreement group's. Held weakly, a world that has been collected reads as "not
    the current one" -- which is the right answer, and the group it belonged to
    still has to be moved aside rather than released.
    """
    built: list[_SpyAgreement] = []

    def build(world_size: int) -> _SpyAgreement:
        agreement = _SpyAgreement()
        built.append(agreement)
        return agreement

    monkeypatch.setattr(stop_agreement_module, "build_stop_agreement", build)
    monkeypatch.setattr(stop_agreement_module, "_agreement", None)
    monkeypatch.setattr(stop_agreement_module, "_agreement_world", None)
    monkeypatch.setattr(stop_agreement_module, "_leaked", [])

    gone = _world("gone")
    first = stop_agreement_module.new_stop_agreement(4, gone)
    del gone

    second = stop_agreement_module.new_stop_agreement(4, _world("current"))
    assert second is not first
    assert stop_agreement_module._leaked == [first]


def test_a_peer_exception_raises_rather_than_stopping_quietly():
    """A peer unwinding is not a preemption, and must not be reported as one.

    The reduced reason carries which it is, so a rank with nothing pending of its
    own learns that it has joined an exception-driven stop and unwinds too --
    exiting non-zero, so torchrun reports a failure rather than a clean stop.
    """
    agreement = _SpyAgreement(reason=StopReason.EXCEPTION.value)
    stop, pending = _stop(agreement)

    with pytest.raises(PeerStopRequested):
        stop.agreed(4)

    assert pending.peer_stop is False, "an exception is not a peer's signal"


def test_the_exit_exchange_happens_only_when_a_rank_originates_one():
    """`close` exchanges iff this rank is originating a stop its peers cannot know.

    A rank leaving because it *observed* a stop would otherwise issue an
    unmatched exchange against a peer that has already left -- and with nothing
    pending of its own it would carry the group's own timeout, so it would sit
    there.
    """
    observed = _SpyAgreement(reason=StopReason.SIGNAL.value)
    stop, pending = _stop(observed)
    assert stop.agreed(3) is True  # sets `_agreed`
    stop.close(stop.next_index())
    assert len(observed.calls) == 1, "an observer must not exchange on the way out"
    pending.close()

    # a clean end of loop exchanges once per boundary and not once more: the scope
    # does not call `close` at all on that path, there being nothing its peers
    # could learn that they are not learning by arriving at the same place
    clean = _SpyAgreement()
    clean_pending = PendingStop(budget=10.0)
    with _cooperative_scope(clean_pending, clean) as clean_stop:
        assert clean_stop.agreed(0) is False
    assert len(clean.calls) == 1, "a clean end of loop must not exchange either"
    clean_pending.close()

    originating = _SpyAgreement(reason=StopReason.EXCEPTION.value)
    stop, pending = _stop(originating, first_index=5)
    assert pending.seconds_remaining() is None
    stop.close(stop.next_index())
    # the index its peers will contribute, not its own lagging counter
    assert originating.calls[0][:2] == (StopReason.EXCEPTION.value, 5)
    # and the raise is a local event, so the exchange carried a short deadline
    assert originating.timeouts[0] < _NO_LOCAL_EVENT_DEADLINE
    pending.close()


def test_close_never_raises_even_when_the_exchange_does():
    """It is called from ``except BaseException: close(...); raise``.

    On the mid-iteration path the exchange is *expected* to time out, so a
    raising `close` would replace the original exception with a timeout and
    defeat the whole point of leaving the traceback intact.
    """

    class _Failing(_SpyAgreement):
        def exchange(self, reason, index, timeout):
            super().exchange(reason, index, timeout)
            raise RuntimeError("Operation timed out!")

    agreement = _Failing()
    stop, pending = _stop(agreement)

    stop.close(stop.next_index())

    assert len(agreement.calls) == 1
    pending.close()


def test_a_give_up_with_no_local_event_dispatches_the_teardown(capfd):
    """The one path that could leave the loop without tearing anything down.

    A rank whose peer never joins gives up on the exchange. If it had recorded a
    signal of its own, `defer_termination`'s dispatch would find that and tear
    down; with **no** local event -- which is every rank the scheduler's signal
    did not reach in time -- there is nothing else on the `PendingStop`, so
    without an explicit record the rank would break out of the loop, tear nothing
    down, and walk into the next collective whose peers have gone.

    The assertion is not on a flag but on the dispatch: `shutdown` ran and a
    `SystemExit` left the deferral, which is what the caller after the loop never
    gets to see.
    """
    events: list[str] = []
    after: list[str] = []
    agreement = _WedgedPeerAgreement()

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(SystemExit) as excinfo:
            with defer_termination(budget=1.0) as pending:
                assert pending.seconds_remaining() is None, "no local event"
                with _cooperative_scope(pending, agreement) as stop:
                    for index in range(10):
                        if stop.agreed(index):
                            break
            after.append("kept going into the next collective")

    assert after == [], "the rank left the loop and carried on"
    assert events == ["shutdown"]
    assert excinfo.value.code == 128 + signal.SIGTERM
    assert pending.requested is False, "no signal ever reached this rank"
    assert pending.peer_stop is True
    stderr = capfd.readouterr().err
    assert len(_marker_lines("agreement-timeout", stderr)) == 1
    assert len(_marker_lines("agreement-abandoned", stderr)) == 1


def test_a_give_up_in_the_loop_entry_exchange_names_no_batch(capfd):
    """The abandonment marker must not report a batch that does not exist.

    Reading one past the last index and subtracting gives ``batch=-1`` when the
    give-up happened in the loop-entry exchange, before any boundary.
    """
    agreement = _WedgedPeerAgreement()
    pending = PendingStop(budget=1.0)

    with pytest.raises(RuntimeError, match="Operation timed out!"):
        with _cooperative_scope(pending, agreement, loop_length=100):
            pass  # never reached

    abandoned = _marker_lines("agreement-abandoned", capfd.readouterr().err)
    assert len(abandoned) == 1
    assert "batch=loop-entry" in abandoned[0]
    pending.close()


def test_an_unverified_torch_degrades_the_bound_rather_than_failing(
    monkeypatch, capfd, caplog
):
    """An unverified torch release must not fail a job, at startup or at use.

    Every distributed job builds an agreement group, including inference and
    evaluator runs that never stop cooperatively, so raising at construction
    breaks work this design does not touch -- and raising at first use would raise
    during a preemption, which is the worst possible moment. So a rank on an
    unverified release passes the same unbounded deadline a healthy rank does, and
    the log says which bound was in force.
    """
    monkeypatch.setattr(stop_agreement_module, "_VERIFIED_TORCH", frozenset())
    monkeypatch.setattr(stop_agreement_module, "_warned_unverified", False)

    # emitted where the group is built, and it warns rather than raising
    with caplog.at_level(logging.WARNING):
        stop_agreement_module.warn_if_timeout_contract_unverified()
        stop_agreement_module.warn_if_timeout_contract_unverified()
    warnings = [
        record for record in caplog.records if "has not been checked" in record.message
    ]
    assert len(warnings) == 1, "once per process, not once per group"
    assert torch.__version__ in warnings[0].getMessage()

    agreement = _SpyAgreement()
    stop, pending = _stop(agreement, budget=10.0)
    pending.request(signal.SIGTERM)
    try:
        assert stop.agreed(4) is True
    finally:
        pending.close()

    assert agreement.timeouts == [_NO_LOCAL_EVENT_DEADLINE]
    bound = _marker_lines("agreement-bound", capfd.readouterr().err)
    assert len(bound) == 1
    assert "bound=group-timeout" in bound[0]
    assert "batch=4" in bound[0]


def test_a_verified_torch_reports_the_bound_it_applied(capfd):
    """The degraded path cannot look like the bounded one in a container log.

    A reader of a rank that sat for the group's whole timeout has to be able to
    tell whether the short bound was in force and failed or was never available,
    and the two produce the same ``stop-agreed`` line. Only a rank with a local
    event of its own emits this, so a healthy loop stays silent.
    """
    agreement = _SpyAgreement()
    stop, pending = _stop(agreement, budget=10.0)

    assert stop.agreed(0) is False
    assert _marker_lines("agreement-bound", capfd.readouterr().err) == []

    pending.request(signal.SIGTERM)
    try:
        assert stop.agreed(1) is True
    finally:
        pending.close()

    bound = _marker_lines("agreement-bound", capfd.readouterr().err)
    assert len(bound) == 1
    assert "bound=budget" in bound[0]
    assert "batch=1" in bound[0]
