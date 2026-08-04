"""Leave a batch loop at the same iteration on every rank.

This is the loop-facing seam, and it joins the two halves below it: the
*deferral* in `fme.core.distributed.shutdown`, which makes a termination signal
record intent rather than tear the backend down, and the *agreement* in
`fme.core.distributed.stop_agreement`, which is how ranks find out about each
other's intent in bounded time. Neither half knows about the other; this module
is the only place that needs both, and it is deliberately the smallest of the
three.

The whole adopter-facing surface is one context manager::

    from fme.core.distributed import Distributed, cooperative_stop

    with Distributed.context():
        with cooperative_stop() as stop:
            for index, batch in enumerate(loader):
                train_on_batch(batch)          # contains the gradient all-reduce
                if stop.agreed(index):
                    break

Leaving the context on a stop performs the same teardown the signal handler would
have performed -- the backend release, the post-shutdown callbacks, the exit code
-- entered from the loop's boundary rather than from wherever the signal landed.

**Every deadline the design chooses lives here**, with the one module that
consumes them, rather than split across the two below it in the layering.

**The mechanism requires the process to exit.** An exchange that is given up on
leaves work in the gloo work queue for good, and no call reclaims the group in
bounded time. A caller that means to stop cooperatively, time out, and then carry
on in the same process is outside what this design can bound.
"""

import contextlib
import logging
from collections.abc import Iterator

from .distributed import Distributed
from .shutdown import PendingStop, StopReason, defer_termination, write_marker
from .stop_agreement import GROUP_TIMEOUT, StopAgreement, is_deadline_expiry

logger = logging.getLogger(__name__)

# Seconds a rank with a local event of its own -- a signal it recorded, or an
# exception it is raising -- will spend waiting for peers that may never arrive.
# Derived by subtraction: the scheduler's grace period is 30-36s from SIGTERM and
# `DEFAULT_TEARDOWN_TIMEOUT` claims 20s of it, so this is what is left.
DEFAULT_STOP_AGREEMENT_BUDGET: float = 10.0

# Floors every deadline a rank with a budget passes, and closes two traps at
# once. torch reads a zero-length timeout as *no* timeout, and the pybind chrono
# caster truncates a `timedelta` to integral milliseconds -- so a rank reaching a
# boundary with 0.4ms of budget left would otherwise wait the group's 30 minutes.
# The size is a decision, not a round number: it is the whole time an already-late
# rank gives its peers to answer, and a measured exchange is 630us at 8 ranks, so
# it clears both the truncation boundary and the 404ms spread with which the
# elastic agent delivers SIGTERM across ranks.
_MIN_DEADLINE: float = 1.0

# What a rank with no local event of its own passes. Derived from the group's own
# timeout rather than restated, so it cannot drift from it. Such a rank has no
# reason to hurry and no standing to declare a peer dead, and a deadline equal to
# the one the gradient all-reduce already carries cannot fire on a job that is not
# already dead -- which is the whole of the false-positive argument.
NO_LOCAL_EVENT_DEADLINE: float = GROUP_TIMEOUT.total_seconds()


class PeerStopRequested(RuntimeError):
    """Raised on a rank whose peer is leaving the loop on an exception."""


class UnequalLoopLength(RuntimeError):
    """Raised on every rank when loop entry finds unequal loop lengths."""


class CooperativeStop:
    """The object a loop polls once per boundary.

    Obtained from `cooperative_stop`, which is what wires it to a deferral and an
    agreement group; constructing one directly is for tests that want to inject a
    scripted agreement.
    """

    def __init__(
        self,
        pending: PendingStop,
        agreement: StopAgreement,
        first_index: int = 0,
    ) -> None:
        self._pending = pending
        self._agreement = agreement
        # a stop has been read out of, *or given up on*, an exchange
        self._agreed = False
        # set by `close`, read by `_local_reason`; never a peer's reason
        self._originating = StopReason.NONE
        self._next_index = first_index

    def _local_reason(self) -> StopReason:
        """Why *this* rank is leaving, if it is. Never a peer's reason.

        It reads state rather than inferring it: ``EXCEPTION`` comes from
        `close`'s assignment, because `agreed` is called from the loop body where
        a rank cannot yet know it is about to raise.

        A rank that recorded a signal reports ``SIGNAL`` even while raising: a
        preemption is what happened to it, and `defer_termination` gives a
        recorded signal the same precedence when it picks an exit code.
        """
        if self._pending.requested:
            return StopReason.SIGNAL
        return self._originating

    def _deadline(self, index: int) -> float:
        """The deadline for one exchange, under the one rule that matters.

        A rank with no local event of its own passes the group's 30 minutes; a
        rank with one passes what is left of its budget, floored.

        A spent budget is not an abandonment and does not cause one. The budget
        bounds how long this rank waits for peers, not how long it may take to
        reach the boundary -- and the usual reason it ran out is that this rank is
        the *late* one, walking out of a multi-second checkpoint write, in which
        case its peers are already blocked in the exchange below and it returns in
        microseconds.
        """
        remaining = self._pending.seconds_remaining()
        if remaining is None:
            return NO_LOCAL_EVENT_DEADLINE
        if remaining <= 0.0:
            write_marker("agreement-expired", batch=str(index))
        return max(remaining, _MIN_DEADLINE)

    def _report(self, reason: StopReason, high: int, low: int) -> None:
        """Emit what one completed exchange revealed.

        The symmetry check is what turns "every rank stopped at the same batch"
        from a claim the log repeats into a fact the mechanism checked. It is
        **diagnostic only**: a desynchronised run is going to hang anyway, and a
        new mechanism unilaterally terminating healthy-looking jobs on a
        self-diagnosis is a worse failure mode than the one it reports.
        """
        if high != low:
            write_marker("index-mismatch", min=str(low), max=str(high))
        if reason is not StopReason.NONE:
            write_marker(
                "stop-agreed",
                batch=str(high),
                world=str(self._agreement.world_size),
                reason=reason.name,
            )

    def assert_equal_loop_length(self, length: int) -> None:
        """One exchange at loop entry, asserting every rank's loop length agrees.

        Equal per-rank batch counts hold by construction today, on both sampler
        paths, so this is insurance against that construction changing rather
        than a live exposure. If it ever did change, the boundary exchanges could
        not detect it: the short rank would leave the loop and exchange nothing,
        and the long rank would sit alone in one more exchange, with the group's
        30 minutes on it and no peer to compare indices against.

        Every rank reads the same reduced values, so every rank raises or none
        does -- which is why this is a raise rather than the diagnostic the
        ±index check is. An unequal loop length is a precondition failure detected
        before any work is done, where continuing is a guaranteed hang.

        Raises:
            UnequalLoopLength: On every rank, if the lengths differ.
        """
        _, high, low = self._agreement.exchange(
            StopReason.NONE.value, length, self._deadline(length)
        )
        if high != low:
            raise UnequalLoopLength(
                f"every rank must iterate the same number of batches, but this "
                f"loop's length is {length} here while ranks reported lengths "
                f"between {low} and {high}. Continuing would hang the first "
                "collective the short rank never reaches."
            )

    def agreed(self, index: int) -> bool:
        """One boundary exchange. Whether this rank should leave the loop now.

        Args:
            index: This rank's iteration index. Correctness rests on this being
                called the same number of times on every rank, not on the value;
                the value is carried so the evidence line and the symmetry
                diagnostic have a number.

        Returns:
            Whether to leave the loop, which is true as soon as *any* rank's
            reason reaches this one -- or as soon as this rank gives up waiting.

        Raises:
            PeerStopRequested: If a peer is leaving on an exception, so that this
                rank unwinds rather than exiting as if it had been preempted.
            RuntimeError: If a peer's process died, closing its socket. Masking
                that as a graceful stop would hide the crash.
        """
        # Recorded before the exchange, not after it, because a `close` or an
        # abandonment marker after a *failed* exchange still has to name the index
        # this rank was working on. On the ordinary path it makes no difference.
        self._next_index = index + 1
        timeout = self._deadline(index)
        try:
            reduced, high, low = self._agreement.exchange(
                self._local_reason().value, index, timeout
            )
        except RuntimeError as err:
            # read out of, *or given up on*, an exchange: a rank that gave up here
            # must not exchange again from `close` against peers already gone
            self._agreed = True
            if not is_deadline_expiry(err):
                # A peer's process exited and closed its socket: a crash, not a
                # wedge. Let it escape rather than reporting a rank that left its
                # peers behind.
                raise
            write_marker("agreement-timeout", batch=str(index), err=repr(str(err)))
            return True
        reason = StopReason(reduced)
        self._report(reason, high, low)
        if reason is StopReason.NONE:
            return False
        self._agreed = True
        if reason is StopReason.EXCEPTION:
            raise PeerStopRequested(
                f"a peer is leaving the loop on an exception at batch {high}"
            )
        # `note_peer_stop` rather than nothing, so that leaving the scope tears
        # this rank down even if the signal was a peer's and never reached here
        self._pending.note_peer_stop()
        return True

    def next_index(self) -> int:
        """The index `close` must contribute in order to match its peers'.

        One past the last index `agreed` was given. A rank raising inside the loop
        body never reached the end of it, so its own counter still holds the
        previous iteration's value while every peer that completed the body has
        incremented -- contributing this rank's last-known index would fire
        ``index-mismatch`` on exactly the path the exception rider covers. Before
        the first `agreed` call this is the ``first_index`` it was constructed
        with.
        """
        return self._next_index

    @property
    def abandoned(self) -> bool:
        """Whether an exchange was given up on, so the group is left behind."""
        return self._agreement.abandoned

    def close(self, reason: StopReason, index: int) -> None:
        """The exit exchange, under the one rule that governs when there is one.

        It exchanges **if and only if this rank is originating a stop its peers
        cannot yet have read out of an exchange** -- exactly ``reason is
        EXCEPTION and not self._agreed``. A rank leaving because it *observed* a
        stop has `_agreed` set, so the stop is already common knowledge and a
        further exchange would be unmatched by the rank that has already left; a
        rank leaving on a clean end of epoch is excluded by the first clause. A
        rank originating an exception-driven stop is the one case its peers cannot
        learn of any other way.

        Args:
            reason: ``EXCEPTION`` if this rank is unwinding, ``NONE`` otherwise --
                including on this rank's own signal, because this branch also
                fires on a clean end of epoch where no signal exists.
            index: The index this rank *would* have reached, i.e. `next_index`.
                A parameter rather than something read off this object, because
                the only value that matches the peers is that one.

        Never raises. On the mid-iteration path the exchange is *expected* to time
        out, and this is called from ``except BaseException: close(...); raise`` --
        so a raising `close` would replace the original exception with a timeout
        and defeat the whole point of leaving the traceback intact.
        """
        # before `_local_reason` is read to build the payload below
        self._originating = reason
        if reason is not StopReason.EXCEPTION or self._agreed:
            return
        # before the exchange it bounds: the raise is a local event on this rank,
        # which is what entitles it to a short deadline
        self._pending.arm_budget()
        try:
            reduced, high, low = self._agreement.exchange(
                self._local_reason().value, index, self._deadline(index)
            )
        except BaseException:
            logger.exception(
                "Failed to tell peers this rank is leaving the loop on an "
                "exception; they may be left in a collective."
            )
            return
        self._report(StopReason(reduced), high, low)


@contextlib.contextmanager
def _cooperative_scope(
    pending: PendingStop,
    agreement: StopAgreement,
    first_index: int = 0,
    loop_length: int | None = None,
) -> Iterator[CooperativeStop]:
    """Everything `cooperative_stop` does except opening the deferral.

    Separate so that multi-rank tests can drive the real exit rules against an
    agreement group of their own. They cannot go through `cooperative_stop`: it
    reaches for the process-wide group, and a test that timed an exchange out on
    that one would leave every later test in the session holding a gloo context
    in an undefined state.
    """
    stop = CooperativeStop(pending, agreement, first_index=first_index)
    try:
        if loop_length is not None:
            stop.assert_equal_loop_length(loop_length)
        try:
            yield stop
        except BaseException:
            # `next_index()`, not this rank's own counter: it never finished the
            # body, so its counter is one behind its peers'
            stop.close(StopReason.EXCEPTION, stop.next_index())
            raise  # the caller's `defer_termination` tears the backend down
        else:
            stop.close(StopReason.NONE, stop.next_index())  # never exchanges
    finally:
        if stop.abandoned:
            # The group is left behind rather than reclaimed, and
            # `destroy_process_group` is silent about it, so this line is the only
            # record. Emitted here because it must precede the teardown, which
            # `defer_termination`'s own `finally` performs next.
            write_marker("agreement-abandoned", batch=str(stop.next_index() - 1))


@contextlib.contextmanager
def cooperative_stop(
    budget: float = DEFAULT_STOP_AGREEMENT_BUDGET,
    first_index: int = 0,
    loop_length: int | None = None,
) -> Iterator[CooperativeStop]:
    """Make the loop this wraps one every rank leaves at the same iteration.

    Args:
        budget: Seconds a rank with a local event of its own will wait for peers.
            Absolute from that event, so a rank cannot accumulate a fresh budget
            at each of several boundaries.
        first_index: The index the loop's first iteration will pass to
            `CooperativeStop.agreed`. It exists only so that a rank raising *in*
            that first iteration contributes the index its peers will contribute;
            a loop over ``enumerate(loader)`` wants the default.
        loop_length: When given, loop entry performs one extra exchange asserting
            that every rank's loop length is equal, raising `UnequalLoopLength` on
            every rank if it is not. ``None`` skips that exchange.

    Yields:
        The object the loop body polls once per boundary.
    """
    dist = Distributed.get_instance()
    # `defer_termination` is the *outer* context, so its `finally` -- which
    # performs the teardown -- runs after `close` has finished exchanging.
    # Nesting them the other way round would tear the backend down and then try
    # to reduce on it.
    with defer_termination(budget=budget) as pending:
        with _cooperative_scope(
            pending,
            dist.stop_agreement(),
            first_index=first_index,
            loop_length=loop_length,
        ) as stop:
            yield stop
