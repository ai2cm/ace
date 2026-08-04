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
Every path that leaves the loop ends that way **where a handler is installed**: a
stop read out of an exchange, a stop given up on, an unequal loop length given up
on at entry, and an exception. A rank that broke out of the loop and carried on
into another collective, whose peers had already left, is the fault this module
exists to prevent.

With no handler installed there is nothing registered to tear down, so leaving the
loop on a stop simply returns -- see ``handle_termination_signals``, which is what
publishes the teardown. A loop reached that way (the whole pytest session, and any
caller using `PendingStop.request` without a handler) must therefore not treat the
loop's end as a *completed* pass over its data; `Trainer.train_one_epoch` does not
increment its epoch counter on a stop for exactly this reason.

**What is covered is the one loop that adopts this, and nothing else.** A signal
arriving during the validation pass, during inline inference, or inside
``run_lr_tuning_trial`` is acted on where it lands, exactly as before this module
existed -- and those paths issue aggregator reductions of their own, so a rank
signalled there still tears the backend down from wherever it was. The one training
loop that adopts this is `fme.core.generics.trainer.Trainer.train_one_epoch`;
`fme/downscaling/train.py`'s `Trainer.train_one_epoch` is a second training loop
that does not. Widening the cover means wrapping those loops too; nothing here does
it for them.

**One window inside the loop is uncovered.** A signal landing after the final
``agreed()`` of an epoch has returned ``False`` finds the ``for`` already ending on
``StopIteration``, so `defer_termination` acts on it unilaterally on the ranks it
reached while ranks it did not reach carry on into the epoch tail -- the shuffle
and the train-evaluation pass -- and strand there. The window is the epoch tail,
which is where the already-accepted epoch-boundary regression lands too, and
`note_peer_stop` exists precisely because a signal may not reach every rank, so
"the elastic agent delivers SIGTERM to every rank within 404ms" is not an argument
this design relies on elsewhere and is not one it relies on here.

**It costs a blocking host-side collective on every iteration** of the loop it
wraps, over a world-wide gloo group, and there is no switch to turn it off. That is
a deliberate omission rather than an oversight: a supported configuration in which
the mechanism is disabled is a supported configuration in which the fabric fault
returns, and adding one remains open to the maintainer. One exchange measured
630 us at 8 gloo ranks -- on an idle workstation, over loopback, with no figure
above 8 ranks and nothing asserting it, so treat that as a lower bound of unknown
looseness rather than a cost. It also removes the run-ahead today's async NCCL
gradient all-reduce allows, capping a rank's host-side lead at one iteration.

**Every deadline the design chooses lives here**, with the one module that
consumes them, rather than split across the two below it in the layering.

**The mechanism requires the process to exit, and after an abandoned exchange it
exits hard.** An exchange that is given up on leaves work in the gloo work queue
for good, and no call reclaims the group in bounded time -- including interpreter
finalization, which joins the worker thread that holds it and so waits for the
wedged peer's socket to close (measured 119.3s against a peer parked 120s on torch
2.7.1, versus 0.05s with ``os._exit``). So the scope asks the deferral for
`PendingStop.require_hard_exit`, and the teardown ends in ``os._exit`` once the
backend is released and the callbacks have run. A caller that means to stop
cooperatively, time out, and then carry on in the same process is outside what this
design can bound.
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

# `index-mismatch` is written once per process rather than once per boundary. The
# condition it reports -- ranks at different iteration indices -- does not un-happen,
# and the run it exists to diagnose is one where every rank would otherwise emit one
# line per batch: tens of thousands of stderr lines per rank on the very run whose
# log a reader is trying to get through.
_reported_index_mismatch: bool = False

# Seconds a rank with a local event of its own -- a signal it recorded, or an
# exception it is raising -- will spend waiting for peers that may never arrive.
#
# Sized against the elastic agent's window, which is what the design is spending:
# torchrun gives the ranks 30s between SIGTERM and SIGKILL (see
# `DEFAULT_TEARDOWN_TIMEOUT`), the teardown claims 20s of it, and what is left is
# for the post-shutdown callbacks -- the restart checkpoint, the reason the
# callbacks exist.
#
# This budget alone does not decide the agreement's share, because `_MIN_DEADLINE`
# floors every deadline. Write T for the seconds between a rank's local event and
# the boundary it next reaches; its exchange then carries max(BUDGET - T,
# _MIN_DEADLINE), so it is done agreeing by
#
#     max(BUDGET, T + _MIN_DEADLINE) = max(5, T + 3)
#
# and the callbacks begin with min(5, 7 - T) seconds of the 30s window left:
#
#   * T <= 2s -- the ordinary case, a rank at a boundary within a batch or two of
#     the signal: the agreement spends 5s exactly as this figure claims, the
#     teardown 20s, and the callbacks get the remaining 5s, a sixth of the window.
#   * T > 2s: the floor overshoots the budget by T - 2 seconds, because a rank that
#     arrives late still gives its peers a whole floor to answer in. A rank walking
#     out of a 10s checkpoint write holds the agreement to t = 13s and leaves the
#     teardown and the callbacks 17s of the window between them, not 25s.
#   * T >= 7s: the callbacks begin at or after the SIGKILL in the worst case, and
#     the restart checkpoint is lost. What keeps T small is that the loop skips a
#     periodic checkpoint write once a stop is pending, so the one routine source
#     of a multi-second T is a write already in flight when the signal landed.
#
# Both terms are ceilings, not reservations: the teardown normally returns in well
# under a second, and a rank whose peers are already waiting returns from the
# exchange in microseconds. Giving a rank longer to wait for peers means taking it
# from the checkpoint, since the teardown timeout is fixed.
DEFAULT_STOP_AGREEMENT_BUDGET: float = 5.0

# Floors every deadline a rank with a budget passes, and closes three traps.
#
# Two are torch's, and 1.0 would close them: a zero-length timeout is read as *no*
# timeout, and the pybind chrono caster truncates a `timedelta` to integral
# milliseconds, so a rank reaching a boundary with 0.4ms of budget left would
# otherwise wait the group's own timeout. Any floor above a millisecond does that,
# and comfortably clears the 404ms spread over which the elastic agent was observed
# to deliver SIGTERM across ranks -- one observation on one configuration, so read it
# as an order of magnitude rather than a bound.
#
# The third trap is the sizing, and it is why this is 3s rather than 1s. The floor
# is the whole time a rank with a spent budget gives its peers to answer -- most
# sharply a rank *originating* an exception, whose peers have no local event of
# their own and so do not know to hurry. Those peers are mid-iteration: they reach
# the boundary only at the end of the batch they are already in, so the floor has to
# exceed a batch. The one figure observed from a production run is ~0.28s/batch, on
# one configuration and with nothing asserting it, which 1s clears by only 3.5x --
# and batch time is configuration-dependent, since `use_gradient_accumulation`
# multiplies the reductions per batch by a stochastic `n_loss_steps`. 3s is an order
# of magnitude over that observation while staying under
# `DEFAULT_STOP_AGREEMENT_BUDGET`, so the budget, not the floor, is still what
# bounds a rank that arrives on time. What it does not clear is a peer inside a
# multi-second checkpoint write; that peer is given up on, and on the signal path
# the loop now skips that write for exactly this reason.
#
# The floor is *not* bounded above by the budget, and that is the one cost of
# choosing it: a rank arriving more than BUDGET - _MIN_DEADLINE = 2s after its own
# local event spends longer agreeing than the budget names. The accounting at
# `DEFAULT_STOP_AGREEMENT_BUDGET` includes that overshoot.
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
        # An exchange *this* object made was given up on. Not read off the
        # agreement, whose flag lives on the process-wide group and so stays set
        # for every later scope in the process -- which would make the
        # `agreement-abandoned` line, whose whole value is being the only record of
        # one event, appear on scopes that abandoned nothing.
        self._abandoned = False
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
        self-diagnosis is a worse failure mode than the one it reports. It is also
        reported once per process; see `_reported_index_mismatch`.
        """
        global _reported_index_mismatch
        if high != low and not _reported_index_mismatch:
            _reported_index_mismatch = True
            write_marker("index-mismatch", min=str(low), max=str(high))
        if reason is not StopReason.NONE:
            write_marker(
                "stop-agreed",
                batch=str(high),
                world=str(self._agreement.world_size),
                reason=reason.name,
            )

    def assert_equal_loop_length(self, length: int) -> bool:
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

        This exchange needs `agreed`'s give-up handling and not only its own
        assertion, because a signal can land between `defer_termination` yielding
        and here -- arming the budget, so this exchange carries
        ``max(remaining, _MIN_DEADLINE)`` while unsignalled peers carry the default
        group's own timeout. The window immediately before it respawns the epoch's
        persistent DataLoader workers, which routinely costs seconds, so a peer more
        than a few seconds from loop entry really does expire this deadline. Without
        the handling below that expiry is a bare ``RuntimeError`` out of a
        preemption: the job reports 1 with a traceback instead of ``128 + signum``.

        Returns:
            Whether a stop was given up on here, in which case the caller must
            **not** run the loop -- its peers are not there, so the body's first
            collective would strand this rank -- and must leave in the way
            `_cooperative_scope` does.

        Raises:
            UnequalLoopLength: On every rank, if the lengths differ.
            RuntimeError: If a peer's process died, closing its socket, exactly as
                in `agreed`.
        """
        try:
            _, high, low = self._agreement.exchange(
                StopReason.NONE.value, length, self._deadline()
            )
        except RuntimeError as err:
            # the same three steps as `agreed`'s give-up branch, for the same
            # reasons: no further exchange, a record of why, and something on the
            # `PendingStop` for `defer_termination`'s dispatch to find
            self._agreed = True
            self._abandoned = True
            if not is_deadline_expiry(err):
                raise
            write_marker(
                "agreement-timeout", batch=self._exchange_label(), err=str(err)
            )
            self._pending.note_peer_stop()
            return True
        if high != low:
            raise UnequalLoopLength(
                f"every rank must iterate the same number of batches, but this "
                f"loop's length is {length} here while ranks reported lengths "
                f"between {low} and {high}. Continuing would hang the first "
                "collective the short rank never reaches."
            )
        return False

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
            self._abandoned = True
            if not is_deadline_expiry(err):
                # A peer's process exited and closed its socket: a crash, not a
                # wedge. Let it escape rather than reporting a rank that left its
                # peers behind.
                raise
            # `write_marker` encodes the message so that it cannot break the
            # space-separated `key=value` contract, which a torch error message
            # otherwise does on the first space it contains
            write_marker("agreement-timeout", batch=str(index), err=str(err))
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
        """Whether *this scope* gave up on an exchange, so the group is left behind.

        Per scope rather than per group, so that the ``agreement-abandoned`` line is
        a record of one event. The agreement's own flag cannot answer this: it lives
        on the process-wide group, so once any loop has abandoned an exchange it
        stays set for the life of the process.
        """
        return self._abandoned

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

        Never raises, and everything after the ``_originating`` assignment is
        guarded for that reason. On the mid-iteration path the exchange is
        *expected* to time out, and this is called from
        ``except BaseException: close(...); raise`` -- so a raising `close` would
        replace the original exception with a timeout and defeat the whole point of
        leaving the traceback intact.
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
            # the operation stays in gloo's work queue, so this scope owes the
            # abandonment line and the hard exit that goes with it
            self._abandoned = True
            logger.exception(
                "Failed to tell peers this rank is leaving the loop on an "
                "exception; they may be left in a collective."
            )
            return
        try:
            self._report(StopReason(reduced), high, low)
        except BaseException:
            # `StopReason(reduced)` raises `ValueError` on a code the enum does not
            # cover. Unreachable while it covers every value a `MAX` over it can
            # produce -- but this method's no-raise guarantee is absolute, and
            # leaving the conversion outside the guard made it the one statement
            # that could replace the caller's exception with a `ValueError`.
            logger.exception("Failed to report the exit exchange.")


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
        if loop_length is not None and stop.assert_equal_loop_length(loop_length):
            # A give-up at loop entry: the loop body must not run, because its first
            # collective would strand this rank against peers that are not there. A
            # context manager cannot skip its own body, so leaving is the only
            # option -- and `SystemExit` is the one that leaves without a traceback
            # and without turning a preemption into a failed job. `defer_termination`
            # treats it as an exit rather than as a propagating exception, so its
            # dispatch tears the backend down, runs the callbacks and exits with the
            # code below, which is `128 + signum` where a signal was recorded and the
            # peer-stop convention otherwise.
            raise SystemExit(pending.exit_code)
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
            # And the process may not be left to finalize: the gloo destructor
            # joins the worker thread that still holds the abandoned operation, so
            # `sys.exit` waits for the wedged peer to die. This is the whole of the
            # design's bound after an abandoned exchange.
            #
            # One path does not honour it, and it is the exception rider: a rank
            # unwinding on an exception is torn down with `exit_process=False` so
            # that its traceback survives, so it returns rather than exiting and may
            # then sit in finalization until its peer dies. The traceback is worth
            # more than the promptness there, and by that point the backend is
            # released and the callbacks have run, so what is lost is only the time.
            pending.require_hard_exit()


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

    Raises:
        UnequalLoopLength: From loop entry, on every rank, when ``loop_length`` is
            given and the lengths differ.
        SystemExit: From loop entry, when the loop-entry exchange is given up on.
            The loop body cannot run in that case, so this is how the scope leaves;
            the teardown and the exit code are the deferral's, as they are for a
            stop agreed at a boundary.
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
