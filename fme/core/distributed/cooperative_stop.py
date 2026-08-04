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
**Every** path that leaves the loop ends that way: a stop read out of an exchange,
a stop given up on, and an exception. A rank that broke out of the loop and carried
on into another collective, whose peers had already left, is the fault this module
exists to prevent.

**What is covered is the one loop that adopts this, and nothing else.** A signal
arriving during the validation pass, during inline inference, or inside
``run_lr_tuning_trial`` is acted on where it lands, exactly as before this module
existed -- and those paths issue aggregator reductions of their own, so a rank
signalled there still tears the backend down from wherever it was. Widening the
cover means wrapping those loops too; nothing here does it for them.

**It costs a blocking host-side collective on every iteration** of the loop it
wraps, over a world-wide gloo group, and there is no switch to turn it off. One
exchange measured 630 us at 8 gloo ranks -- on an idle workstation, over loopback,
with no figure above 8 ranks and nothing asserting it, so treat that as a lower
bound of unknown looseness rather than a cost. It also removes the run-ahead
today's async NCCL gradient all-reduce allows, capping a rank's host-side lead at
one iteration.

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
from .stop_agreement import (
    StopAgreement,
    default_group_timeout,
    is_deadline_expiry,
    timeout_contract_verified,
)

logger = logging.getLogger(__name__)

# Seconds a rank with a local event of its own -- a signal it recorded, or an
# exception it is raising -- will spend waiting for peers that may never arrive.
#
# Sized against the elastic agent's window, which is what the design is spending:
# torchrun gives the ranks 30s between SIGTERM and SIGKILL (see
# `DEFAULT_TEARDOWN_TIMEOUT`), this claims 5s of it and the teardown 20s, leaving
# the post-shutdown callbacks -- the restart checkpoint, the reason the callbacks
# exist -- the remaining 5s, a sixth of the window. Giving a rank longer to wait
# for peers means taking it from the checkpoint, since the teardown timeout is
# fixed.
DEFAULT_STOP_AGREEMENT_BUDGET: float = 5.0

# Floors every deadline a rank with a budget passes, and closes three traps.
#
# Two are torch's, and 1.0 would close them: a zero-length timeout is read as *no*
# timeout, and the pybind chrono caster truncates a `timedelta` to integral
# milliseconds, so a rank reaching a boundary with 0.4ms of budget left would
# otherwise wait the group's own timeout. Any floor above a millisecond does that,
# and comfortably clears the 404ms spread with which the elastic agent delivers
# SIGTERM across ranks.
#
# The third trap is the sizing, and it is why this is 3s rather than 1s. The floor
# is the whole time a rank with a spent budget gives its peers to answer -- most
# sharply a rank *originating* an exception, whose peers have no local event of
# their own and so do not know to hurry. Those peers are mid-iteration: they reach
# the boundary only at the end of the batch they are already in, so the floor has to
# exceed a batch. The one figure measured from production runs is ~0.28s/batch,
# which 1s clears by only 3.5x -- and batch time is configuration-dependent, since
# `use_gradient_accumulation` multiplies the reductions per batch by a stochastic
# `n_loss_steps`. 3s is an order of magnitude over the measured figure while staying
# under `DEFAULT_STOP_AGREEMENT_BUDGET`, so the budget, not the floor, is still what
# bounds a rank that arrives on time. What it does not clear is a peer inside a
# multi-second checkpoint write; that peer is given up on, and on the signal path
# the loop now skips that write for exactly this reason.
_MIN_DEADLINE: float = 3.0


def no_local_event_deadline() -> float:
    """Seconds a rank with no local event of its own passes.

    Such a rank has no reason to hurry and no standing to declare a peer dead, so
    it passes the deadline the gradient all-reduce already carries: a deadline
    equal to that cannot fire on a job that is not already dead, which is the whole
    of the false-positive argument.

    That figure is read off the default group rather than restated here, because
    the two backends configure it differently -- see `default_group_timeout`. It is
    also what a rank with a local event falls back to on a torch release whose
    timeout behaviours were never checked; see `timeout_contract_verified`.
    """
    return default_group_timeout()


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
        # The index of the exchange this object is in or last made, for the
        # abandonment marker. `None` until the first one, so that a give-up in the
        # loop-entry exchange is not reported against a batch index that does not
        # exist.
        self._exchange_index: int | None = None
        # Both read once here rather than per boundary. The first is a torch
        # version check and the second walks torch's group bookkeeping, and neither
        # can change inside one loop -- while `agreed` runs tens of thousands of
        # times per epoch.
        self._bounded = timeout_contract_verified()
        self._no_local_event_deadline = no_local_event_deadline()
        # `agreement-bound` is written once per rank per loop, at the first
        # boundary a rank passes with a local event of its own
        self._reported_bound = False

    @property
    def pending(self) -> PendingStop:
        """The deferral's record of a stop this rank has recorded but not acted on.

        Exposed for two things the loop cannot otherwise do. It can **skip work**
        once a stop is pending -- a multi-GB periodic checkpoint write above all,
        which would otherwise delay the agreed stop by the whole write, and which
        is redundant with the restart checkpoint about to be taken at the very
        boundary the ranks are agreeing on. And it makes `PendingStop.request`
        reachable, so a caller can stop for a reason of its own -- a wall-clock
        budget, say -- from the surface it actually holds.
        """
        return self._pending

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

    def _deadline(self) -> float:
        """The deadline for one exchange, under the one rule that matters.

        A rank with no local event of its own passes the deadline the gradient
        all-reduce already carries; a rank with one passes what is left of its
        budget, floored -- unless the installed torch is one whose timeout
        behaviours were never read, in which case it passes the same unbounded
        figure a healthy rank does, and says so.

        A spent budget is not an abandonment and does not cause one. The budget
        bounds how long this rank waits for peers, not how long it may take to
        reach the boundary -- and the usual reason it ran out is that this rank is
        the *late* one, walking out of a multi-second checkpoint write, in which
        case its peers are already blocked in the exchange below and it returns in
        microseconds.
        """
        remaining = self._pending.seconds_remaining()
        if remaining is None:
            return self._no_local_event_deadline
        if not self._bounded:
            self._report_bound("group-timeout", self._no_local_event_deadline)
            return self._no_local_event_deadline
        deadline = max(remaining, _MIN_DEADLINE)
        self._report_bound("budget", deadline)
        return deadline

    def _report_bound(self, bound: str, deadline: float) -> None:
        """Record which bound is in force, once per loop, on the rank it binds.

        Without this the degraded path is indistinguishable from the bounded one in
        a container log: both produce the same ``stop-agreed`` line, and the reader
        of a rank that sat for the default group's whole timeout has no way to tell
        whether the design's short bound was in force and failed or was never
        available at all. Only ranks with a local event of their own emit it, so a
        healthy loop is silent.
        """
        if self._reported_bound:
            return
        self._reported_bound = True
        write_marker(
            "agreement-bound",
            batch=self._exchange_label(),
            bound=bound,
            seconds=f"{deadline:.1f}",
        )

    def _exchange_label(self) -> str:
        """Which exchange a marker is about, for the two that are not per boundary.

        A string, because the loop-entry exchange contributes a loop *length* rather
        than a batch index and so has no batch to name. Reading one past the last
        index and subtracting, which is what an earlier version did, reported
        ``batch=-1`` for exactly that case.
        """
        if self._exchange_index is None:
            return "loop-entry"
        return str(self._exchange_index)

    def _boundary_deadline(self, index: int) -> float:
        """`_deadline`, plus the marker a spent budget owes at a batch boundary.

        Separate from `_deadline` because the marker's ``batch=`` field has to be a
        batch index, and `assert_equal_loop_length`'s exchange has a loop *length*
        to contribute instead -- so calling this from there would label a length as
        a batch.
        """
        remaining = self._pending.seconds_remaining()
        if remaining is not None and remaining <= 0.0:
            write_marker("agreement-expired", batch=str(index))
        return self._deadline()

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
        and the long rank would sit alone in one more exchange, carrying
        `no_local_event_deadline` and with no peer to compare indices against.

        Every rank reads the same reduced values, so every rank raises or none
        does -- which is why this is a raise rather than the diagnostic the
        ±index check is. An unequal loop length is a precondition failure detected
        before any work is done, where continuing is a guaranteed hang.

        Raises:
            UnequalLoopLength: On every rank, if the lengths differ.
        """
        _, high, low = self._agreement.exchange(
            StopReason.NONE.value, length, self._deadline()
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
        self._exchange_index = index
        timeout = self._boundary_deadline(index)
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
            # Not optional, and the one thing that keeps this branch from
            # reintroducing the fault. A rank that gave up with no local event of
            # its own has nothing else on the `PendingStop` for
            # `defer_termination`'s dispatch to find, so without this it would
            # leave the loop, tear nothing down, and walk into the next collective
            # -- whose peers have gone -- holding an abandoned gloo group, to be
            # SIGKILLed there with its communicators open.
            self._pending.note_peer_stop()
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

    def abandoned_at(self) -> str:
        """Which exchange was given up on, for the abandonment marker."""
        return self._exchange_label()

    def close(self, index: int) -> None:
        """The exit exchange, for a rank leaving the loop on an exception.

        It exchanges **if and only if this rank is originating a stop its peers
        cannot yet have read out of an exchange** -- exactly ``not self._agreed``.
        A rank leaving because it *observed* a stop has `_agreed` set, so the stop
        is already common knowledge and a further exchange would be unmatched by
        the rank that has already left. A rank originating an exception-driven stop
        is the one case its peers cannot learn of any other way, and a rank leaving
        on a clean end of epoch does not call this at all -- there is nothing for
        its peers to learn and they are all arriving at the same place anyway.

        Args:
            index: The index this rank *would* have reached, i.e. `next_index`.
                A parameter rather than something read off this object, because
                the only value that matches the peers is that one.

        Never raises. On the mid-iteration path the exchange is *expected* to time
        out, and this is called from ``except BaseException: close(...); raise`` --
        so a raising `close` would replace the original exception with a timeout
        and defeat the whole point of leaving the traceback intact.
        """
        # before `_local_reason` is read to build the payload below
        self._originating = StopReason.EXCEPTION
        if self._agreed:
            return
        self._exchange_index = index
        # before the exchange it bounds: the raise is a local event on this rank,
        # which is what entitles it to a short deadline
        self._pending.arm_budget()
        try:
            reduced, high, low = self._agreement.exchange(
                self._local_reason().value, index, self._boundary_deadline(index)
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
            stop.close(stop.next_index())
            raise  # the caller's `defer_termination` tears the backend down
    finally:
        if stop.abandoned:
            # The group is left behind rather than reclaimed, and
            # `destroy_process_group` is silent about it, so this line is the only
            # record. Emitted here because it must precede the teardown, which
            # `defer_termination`'s own `finally` performs next.
            write_marker("agreement-abandoned", batch=stop.abandoned_at())


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
        The object the loop body polls once per boundary. Its `pending` property
        is the deferral's own record, so the loop can also skip work once a stop
        is pending rather than only stopping at the next boundary.
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
