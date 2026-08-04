"""Tear the distributed backend down cleanly when a job is preempted.

Schedulers preempt jobs by sending SIGTERM and escalating to SIGKILL once a
grace period expires. A rank that is killed with its NCCL communicators still
open drops its NVLink peers abruptly, which faults the surviving ranks with
``CUDA error: Invalid access of peer GPU memory over nvlink or a hardware
error``. That raises SXid errors on the fabric, so the GPUs need a reboot and
the cluster cordons the node.

``destroy_process_group`` is a collective, so avoiding that outcome is a matter
of every rank reaching it at the same time. Anything slow -- writing a restart
checkpoint, in particular -- has to happen *after* the teardown, not before it,
or the ranks that got there first sit in the collective until the scheduler
kills them.

The handler runs only when the main thread returns to the interpreter, so a
rank whose main thread is blocked in a C-level call (a collective's stream
sync, most importantly) when the signal arrives never starts the teardown and
rides to the scheduler's SIGKILL. The watchdog thread bounds the collective
once it has begun; it cannot start one. Likewise the instants before the
handler is installed -- ``init_process_group`` itself, inside
``Distributed.context()`` entry -- remain unprotected.

The watchdog covers the collective and stops there. Whatever runs afterwards --
the restart checkpoint above all -- is deliberately unbounded: by then the
peers are already out of the collective, so a deadline could only truncate the
write it was meant to make room for.

Tearing down from inside the handler is right everywhere except inside a batch
loop, where the ranks share a rendezvous they could agree to leave at instead.
So this module owns three things: *deferral* -- ``defer_termination`` makes the
handler record intent rather than act, and performs the teardown when the scope
it guards is left -- the evidence marker writer ``write_marker``, whose lines
are the only per-rank record that survives non-root ranks' log level, and the
watchdog, which now aborts the local communicator before its hard exit.
Deciding *when* a deferred stop is acted on is the caller's, not this module's:
nothing here knows about process groups or collectives, which is what keeps the
signal handling testable single-rank and in process.
"""

import contextlib
import enum
import logging
import os
import signal
import sys
import threading
import time
import types
from collections.abc import Callable, Generator

from fme.core.device import in_dataloader_worker

logger = logging.getLogger(__name__)

TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)

# Bounds the collective teardown and nothing after it, per the module docstring.
#
# torchrun's elastic agent, not the scheduler, is the binding constraint. Beaker
# allows 5 minutes between SIGTERM and SIGKILL [1], but the agent gives the
# ranks 30s: `PContext.close` defaults to `timeout=30` and
# `LocalElasticAgent._shutdown` calls it without an argument
# (torch/distributed/elastic/agent/server/local_elastic_agent.py:372), then
# SIGKILLs whoever is still alive. That 30s is one budget shared by every rank
# rather than 30s each, so a slow rank spends its peers' allowance too. 20s
# leaves 10s of margin beneath it.
#
# [1]: https://beaker-docs.apps.allenai.org/scheduling/interruption.html#automated-preemption
DEFAULT_TEARDOWN_TIMEOUT = 20.0

# Seconds between the watchdog aborting the local communicator and exiting the
# process anyway. Deliberately shorter than the callbacks it sits over, which is
# only safe because `_Deadlines` stands the backstop down by a flag rather than
# by `Timer.cancel()`: a successful abort releases the main thread at once, and
# exiting out from under the checkpoint write it then starts would be worse than
# not aborting at all.
_ABORT_BACKSTOP: float = 5.0

# The exit code the watchdog and the escalating phases use when a rank is leaving
# on an exception. Nothing calls `sys.exit` with it: the exception path returns so
# that the original traceback reaches the interpreter unmasked.
_EXCEPTION_EXIT_CODE: int = 1

# A peer's stop carries no signal number, so a rank leaving because a peer was
# signalled reports SIGTERM's conventional code.
_PEER_STOP_EXIT_CODE: int = 128 + int(signal.SIGTERM)

_post_shutdown_callbacks: list[Callable[[], None]] = []

# The deferral registry. `None` means a signal is acted on immediately, which is
# the right thing everywhere but inside a scope that has a rendezvous of its own.
_pending_stop: "PendingStop | None" = None

# Published by `handle_termination_signals` so that a deferral, which has no
# access to that function's locals, can run the same teardown the handler would
# have run. `None` means no handler is installed and there is nothing to tear
# down -- the state every pytest session is in.
_terminate: Callable[..., None] | None = None

# The pid the handler was installed in, so that every marker line can be read
# against it: a real rank has `pid == installed_pid`, while a fork-started
# DataLoader worker sharing its parent's RANK does not.
_installed_pid: int | None = None


class StopReason(enum.Enum):
    """Why a rank is leaving a loop, and the reason code an agreement carries.

    The values are what the agreement's payload transports, so ``MAX`` over them
    is both a logical OR and a choice of the more serious reason: if any rank
    recorded a signal every rank reads at least ``1``, and if any rank is
    unwinding on an exception every rank reads ``2``.

    It lives here rather than beside the agreement so that the agreement stays a
    torch-only leaf with no dependency on the signal handling: what crosses the
    wire is the integer value, not the enum.
    """

    NONE = 0
    SIGNAL = 1
    EXCEPTION = 2


class _Phase(enum.Enum):
    """How far the teardown has got, which is what a repeated signal turns on."""

    RUNNING = enum.auto()
    STOP_REQUESTED = enum.auto()
    COLLECTIVE = enum.auto()
    CALLBACKS = enum.auto()
    COMPLETE = enum.auto()


def add_post_shutdown_callback(callback: Callable[[], None]) -> None:
    """Register work to run on termination, once the process group is gone.

    Callbacks run in registration order and must not use collectives, the
    process group having already been destroyed.

    They are not bounded by the teardown deadline, but they are still
    best-effort: torchrun SIGKILLs the rank about 30s after the signal reaches
    the agent, whatever it is doing (see `DEFAULT_TEARDOWN_TIMEOUT`). Register
    the most valuable work first.

    Registering the same callback twice runs it twice; a caller that may be
    constructed more than once per process should guard its own registration.
    """
    _post_shutdown_callbacks.append(callback)


def clear_post_shutdown_callbacks() -> None:
    """Discard every registered callback.

    The registry is process-global, so tests that build objects which register
    callbacks need to reset it between cases.
    """
    _post_shutdown_callbacks.clear()


def clear_pending_stop() -> None:
    """Discard any registered deferral.

    The counterpart of `clear_post_shutdown_callbacks`, and needed for the same
    reason: the registry is process-global, and a test session that enters
    `Distributed.context(handle_signals=False)` installs no handler, so
    `handle_termination_signals`' own clear-on-exit never runs. A deferral left
    behind by a failed test would otherwise make the next test's signal deferred
    to a scope nobody polls.
    """
    global _pending_stop
    _pending_stop = None


def write_marker(event: str, **fields: str) -> None:
    """Write one machine-readable evidence line to stderr. Never raises.

    ``logging`` cannot carry this: `fme/core/logging_utils.py` puts every
    non-root rank at ERROR, so an INFO line from ranks 1-7 never reaches the
    container log -- and the two things a reader needs are per-rank and negative
    (which rank failed to reach the stopping point, which rank's teardown did not
    return), so a rank-0 summary cannot serve. ``os.write`` also cannot deadlock
    on the logging lock, which matters because these lines are emitted from
    signal handlers and timer threads, and a single short write is atomic on a
    pipe, so several ranks' lines cannot interleave.

    ``installed_pid`` comes from the module-level `_installed_pid` rather than
    from a parameter, so that callers with no access to the handler's locals --
    timer threads, and the loop-facing layer above this module -- emit lines one
    parser stays valid for. It is ``?`` where no handler is installed.
    """
    try:
        rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "?"
        local_rank = (
            os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or "?"
        )
        installed = "?" if _installed_pid is None else str(_installed_pid)
        parts = [
            f"fme-stop:{event}",
            f"rank={rank}",
            f"local_rank={local_rank}",
            f"pid={os.getpid()}",
            f"installed_pid={installed}",
            f"wall={time.time():.6f}",
            f"mono={time.monotonic():.6f}",
            *(f"{name}={value}" for name, value in fields.items()),
        ]
        os.write(2, (" ".join(parts) + "\n").encode())
    except BaseException:
        # A marker is evidence about a teardown, so it may not become a second
        # failure during one: a closed or full stderr must not raise out of a
        # signal handler or a timer thread.
        pass


def _warn_after(timeout: float, message: str) -> threading.Timer:
    """Report a deferral that is taking too long, without acting on it.

    The only lever a timer thread has over a main thread wedged in a collective
    is ``os._exit``, which is the fabric fault this mechanism exists to avoid;
    taking it here would trade a rank we have merely lost track of for a node the
    cluster cordons. So this only writes, which is still strictly better than
    today, where a rank that rides to SIGKILL produces no line at all.

    ``os.write`` rather than the logger, for the reason given in
    `_hard_exit_after`.
    """

    def warn() -> None:
        write_marker("deferral-overrun", since=f"{timeout:.0f}s")
        os.write(2, message.encode())

    timer = threading.Timer(timeout, warn)
    timer.daemon = True
    timer.start()
    return timer


class PendingStop:
    """A stop recorded but not yet acted on. One per `defer_termination`.

    The budget it holds is what bounds a rank that is leaving: it is absolute
    from the first local event -- a signal recorded here, or an exception the
    caller is raising -- so a caller cannot accumulate a fresh budget at each of
    several rendezvous. A rank with no local event of its own arms nothing and
    `seconds_remaining` stays ``None``, which is what keeps a healthy rank from
    ever starting a clock it could then be killed by.
    """

    def __init__(self, budget: float) -> None:
        self._budget = budget
        # a signal was recorded on *this* rank
        self.requested = False
        # a peer's stop was reported to this rank by its caller
        self.peer_stop = False
        # `128 + signum`, meaningful once `requested`; until then the convention
        # for a stop that carries no signal number of its own
        self.exit_code = _PEER_STOP_EXIT_CODE
        self._deadline: float | None = None
        self._overrun: threading.Timer | None = None

    def request(self, signum: int) -> None:
        """From the handler: record the signal and arm the budget.

        Public so that a caller can stop for a reason of its own -- a wall-clock
        budget, say -- and so that multi-rank tests can drive the pending state
        without sending a signal into a live pytest session.
        """
        self.requested = True
        self.exit_code = 128 + signum
        self.arm_budget()
        if self._overrun is None:
            # Starting a timer takes `threading`'s own locks, so a signal landing
            # while the main thread holds one can deadlock -- the same hazard
            # `_hard_exit_after` already carries, since today's handler starts a
            # timer too. What is new is that this one fires on every deferral
            # rather than only when a teardown is under way.
            overrun = 2.0 * self._budget
            self._overrun = _warn_after(
                overrun,
                f"A termination signal has been pending for {overrun:.0f}s "
                "without reaching a stopping point; this rank may be killed with "
                "its communicators open. GPUs on this node may need to be "
                "reset.\n",
            )

    def arm_budget(self) -> None:
        """Arm the budget with no signal, for a caller raising an exception.

        Idempotent, so the budget stays absolute from the first local event.
        """
        if self._deadline is None:
            self._deadline = time.monotonic() + self._budget

    def note_peer_stop(self) -> None:
        """Record that a peer is stopping, so that leaving the scope tears down."""
        self.peer_stop = True

    def seconds_remaining(self) -> float | None:
        """Seconds left in the budget, or ``None`` when no budget is armed.

        May be negative: a budget can be wholly spent before the rank reaches the
        rendezvous it bounds, and arriving late is not the same as being
        abandoned. What the caller does with a spent budget is the caller's.
        """
        if self._deadline is None:
            return None
        return self._deadline - time.monotonic()

    def close(self) -> None:
        """Cancel the diagnostic timer. Never raises."""
        overrun = self._overrun
        self._overrun = None
        if overrun is not None:
            # `Timer.cancel` only sets an event, so it cannot raise
            overrun.cancel()


@contextlib.contextmanager
def defer_termination(budget: float) -> Generator[PendingStop, None, None]:
    """Record a termination signal instead of acting on it, for this scope.

    Inside this context a signal sets the yielded `PendingStop` and returns, so
    the caller can leave whatever it is doing at a point of its own choosing --
    a batch boundary its peers also reach, rather than wherever the signal
    happened to land. Leaving the context then runs the same teardown the
    handler would have run: `shutdown`, the post-shutdown callbacks, and an exit.
    Outside it, nothing changes and a signal is acted on immediately, which is
    right because outside a loop there is no future rendezvous to wait for.

    Well-defined with no handler installed: it yields a `PendingStop` that simply
    never becomes requested, and finds nothing to tear down on the way out.

    Args:
        budget: Seconds a rank with a local event of its own is prepared to spend
            before it stops waiting. Required rather than defaulted, because the
            layer that decides deadlines sits above this module.

    Raises:
        RuntimeError: If a deferral is already registered. The registry is
            process-global and only one scope can own the teardown.
    """
    global _pending_stop
    if threading.current_thread() is not threading.main_thread():
        # a thread cannot own the process's disposition, so no signal is recorded
        # here and nothing would ever poll a registration
        yield PendingStop(budget)
        return
    if in_dataloader_worker():
        # The DataLoader owns its workers' lifecycle. A fork-started worker
        # inherits both this registry and the handler, so registering here would
        # let a worker's copy of the teardown run against a fork-inherited
        # process group; `handle`'s pid guard is the other half of this.
        yield PendingStop(budget)
        return
    if _pending_stop is not None:
        raise RuntimeError("Nested defer_termination() is not supported.")
    pending = PendingStop(budget)
    _pending_stop = pending
    raising = False
    try:
        yield pending
    except BaseException:
        raising = True
        raise
    finally:
        # Clearing the registry *first* closes the window between the caller's
        # last poll and here, in which a signal could otherwise be recorded by a
        # deferral nobody will poll again: from now on a signal takes the
        # immediate path, and its `sys.exit` propagates out of this `finally`
        # rather than letting the teardown below run twice.
        _pending_stop = None
        pending.close()
        if _terminate is None:
            pass  # no handler installed, so there is nothing to tear down
        elif pending.requested:
            # A recorded signal takes precedence over an exception deliberately:
            # a preemption is not a failure, and `128 + signum` is what the
            # scheduler and torchrun expect from a rank asked to stop. The cost
            # is that a rank which recorded a signal *and* raised loses the
            # raise's exit code.
            _terminate(pending.exit_code)
        elif pending.peer_stop:
            _terminate(_PEER_STOP_EXIT_CODE)
        elif raising:
            # Tear down and run the callbacks, then let the exception go on
            # propagating: without this the rank would exit with its
            # communicators open, and with a `sys.exit` here the traceback that
            # brought us here would be replaced by an exit code.
            _terminate(_EXCEPTION_EXIT_CODE, exit_process=False)


class _Deadlines:
    """The watchdog's deadline and its abort backstop, stood down together.

    The flag, not ``Timer.cancel()``, is what stands the backstop down: a
    successful abort releases the main thread immediately, so a cancel can arrive
    while the watchdog thread is still between its abort and its
    ``backstop.start()``, and ``cancel()`` on an unstarted timer does nothing.
    The backstop would then start unopposed and exit the process out from under a
    main thread that had already been released and was writing a checkpoint.

    Constructed empty and given its timers afterwards, because both timers'
    callbacks close over this object; taking them in ``__init__`` would be
    circular.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._deadline: threading.Timer | None = None
        self._backstop: threading.Timer | None = None

    def set_timers(self, deadline: threading.Timer, backstop: threading.Timer) -> None:
        with self._lock:
            self._deadline = deadline
            self._backstop = backstop

    def arm_backstop(self) -> bool:
        """Start the backstop, unless a cancel has already landed."""
        with self._lock:
            if self._cancelled or self._backstop is None:
                return False
            self._backstop.start()
            return True

    def backstop_may_exit(self) -> bool:
        """Whether the process may still be exited, i.e. nothing stood us down."""
        with self._lock:
            return not self._cancelled

    def cancel(self) -> None:
        """Stand both timers down, whether or not either has been started."""
        with self._lock:
            self._cancelled = True
            for timer in (self._deadline, self._backstop):
                if timer is not None:
                    timer.cancel()


def _hard_exit_after(
    timeout: float,
    exit_code: int,
    abort: Callable[[], None] | None = None,
) -> _Deadlines:
    """Abort the local communicator, then exit, if the teardown overruns.

    A blocked collective does not return to the interpreter, so a Python signal
    handler or ``signal.alarm`` would never run; only a separate thread can
    enforce the deadline.

    Aborting first converts a hard exit taken with this rank's communicators open
    into an ordered abort of those communicators followed by an exit, and it may
    release this rank's own main thread from the collective -- in which case the
    teardown continues gracefully and the backstop stands down. Whether an
    aborted communicator leaves the fabric better off than one dropped by
    ``os._exit`` is unverified, and the abort may fault the fabric sooner rather
    than later, so ``os._exit`` remains as a named backstop and this is reversible
    on the first GPU evidence.
    """

    def give_up() -> None:
        # Arming *before* the abort, and refusing to arm once a cancel has landed,
        # is the whole of the race: an abort that releases the main thread lets it
        # cancel us, and a cancel must never find an unstarted timer.
        if not deadlines.arm_backstop():
            # `shutdown` returned as this thread woke: the peers are out, so
            # there is nothing to abort and nothing to warn about
            return
        # the marker before the abort, so a log tells the two outcomes apart
        write_marker("watchdog-abort", timeout=f"{timeout:.0f}")
        # `os.write` rather than the logger: a handler holds its lock across an
        # emit, so if the signal arrived inside that window the main thread holds
        # it and is now wedged below. Acquiring it from this thread would block
        # for good, leaving the rank to be SIGKILLed.
        os.write(
            2,
            f"Distributed shutdown did not complete within {timeout:.0f}s, "
            "aborting the local communicator. GPUs on this node may need to be "
            "reset.\n".encode(),
        )
        if abort is not None:
            try:
                abort()
            except BaseException:
                # nothing is left to try, and raising here would only lose the
                # backstop that follows
                logger.exception("Failed to abort the local communicator.")

    def backstop_exit() -> None:
        # Named backstop, not the primary action: if the abort did not release the
        # main thread, holding the rank until the scheduler's SIGKILL would kill
        # it with its communicators intact -- the outcome this module exists to
        # avoid.
        if not deadlines.backstop_may_exit():
            # `shutdown` returned after the abort: the main thread is running the
            # post-shutdown callbacks and must not be exited out from under them
            return
        os._exit(exit_code)

    # the two closures above read `deadlines` by name, which is legal so long as
    # it is bound before either runs -- i.e. before the deadline timer starts
    deadlines = _Deadlines()
    deadline_timer = threading.Timer(timeout, give_up)
    backstop_timer = threading.Timer(_ABORT_BACKSTOP, backstop_exit)
    deadlines.set_timers(deadline_timer, backstop_timer)
    deadline_timer.daemon = True
    backstop_timer.daemon = True
    deadline_timer.start()
    return deadlines


def _shut_down_backend(shutdown: Callable[[], None], deadline: _Deadlines) -> None:
    """Release the backend, then stand the watchdog down.

    Cancelling here rather than after the callbacks is the point: the moment
    `shutdown` returns the peers are out of the collective, and a deadline left
    armed past that could only cap the restart checkpoint.

    This may not raise: the callbacks are worth attempting even if the backend
    could not be released.
    """
    try:
        shutdown()
    except BaseException:
        logger.exception("Failed to shut down the distributed backend.")
    finally:
        deadline.cancel()


def _run_post_shutdown_callbacks() -> None:
    """Run the best-effort callbacks in registration order.

    Nothing here may raise: every remaining callback is worth attempting even if
    an earlier one failed.
    """
    for callback in _post_shutdown_callbacks:
        try:
            callback()
        except BaseException:
            # BaseException so a callback calling sys.exit() cannot skip the
            # remaining callbacks or hijack the exit code
            logger.exception("Post-shutdown callback %r failed.", callback)


@contextlib.contextmanager
def handle_termination_signals(
    shutdown: Callable[[], None],
    teardown_timeout: float = DEFAULT_TEARDOWN_TIMEOUT,
    abort: Callable[[], None] | None = None,
) -> Generator[None, None, None]:
    """Shut the distributed backend down before exiting on SIGTERM or SIGINT.

    Args:
        shutdown: Tears the distributed backend down. Called before any
            callback registered with `add_post_shutdown_callback`, so that
            every rank reaches the collective teardown together.
        teardown_timeout: Seconds to allow `shutdown` before exiting
            regardless. It does not bound the callbacks that follow; those are
            left to run against the scheduler's clock.
        abort: Aborts this rank's communicators, called by the watchdog before
            its hard exit. A callable rather than a process group, so that this
            module needs no torch import and every claim about signal handling
            stays testable without a launcher.
    """
    if threading.current_thread() is not threading.main_thread():
        # only the main thread may install handlers; a thread shares the
        # process disposition its main thread installed
        yield
        return
    if in_dataloader_worker():
        # A spawn- or forkserver-started worker enters this context only to learn
        # its rank, and the DataLoader owns its lifecycle: exiting out from under
        # it is not ours to do. Fork-started workers never come through here --
        # they inherit the handler, which the pid guard below catches.
        yield
        return

    installed_pid = os.getpid()
    phase = _Phase.RUNNING

    def terminate(exit_code: int, *, exit_process: bool = True) -> None:
        """Release the backend, run the callbacks, and exit.

        The only teardown implementation there is, reached either from the
        handler or from a deferral being left, so the phase transitions are
        identical whichever entry ran.

        `exit_process=False` is for a caller that is already propagating an
        exception: everything happens except the final `sys.exit`, so the
        exception goes on propagating with its traceback intact.
        """
        nonlocal phase
        phase = _Phase.COLLECTIVE
        started = time.monotonic()
        # armed inside the argument expression so that `_shut_down_backend`'s
        # `finally` cancels the exact deadline that bounds it
        _shut_down_backend(
            shutdown, _hard_exit_after(teardown_timeout, exit_code, abort)
        )
        write_marker("shutdown-returned", elapsed=f"{time.monotonic() - started:.2f}s")
        phase = _Phase.CALLBACKS
        try:
            _run_post_shutdown_callbacks()
        finally:
            phase = _Phase.COMPLETE
        if exit_process:
            sys.exit(exit_code)

    def handle(signum: int, frame: types.FrameType | None) -> None:
        nonlocal phase
        exit_code = 128 + signum
        if os.getpid() != installed_pid:
            # A forked child inherited this handler: the default DataLoader start
            # method is fork and the scheduler signals the whole process group, so
            # every worker would otherwise tear down a fork-inherited process
            # group and re-run the parent's callbacks, racing its checkpoint
            # write. Die as this process would have without the inheritance.
            #
            # This guard stays *first*, before the deferral branch below: a worker
            # that recorded intent and returned would be left alive and deaf,
            # having neither died as it would have nor gained anyone to poll it.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        if phase is _Phase.STOP_REQUESTED:
            # In production a second SIGTERM is the norm rather than an
            # escalation: the scheduler signals the container's process group and
            # torchrun's agent then signals every rank again. Tearing down here
            # would therefore defeat the deferral on essentially every real
            # preemption. The pending stop is bounded by the caller's own budget,
            # so the reasoning at `_Phase.COLLECTIVE` below applies verbatim.
            write_marker("signal-ignored", signal=signal.Signals(signum).name)
            logger.info(
                "Received %s while a stop is already pending; ignoring.",
                signal.Signals(signum).name,
            )
            return
        if phase is _Phase.COLLECTIVE:
            # a repeated Ctrl-C, or both the scheduler and torchrun signalling.
            # The first handler owns the collective and the deadline already
            # bounds it, so restarting it here would only cost us the callbacks.
            logger.info(
                "Received %s while the backend is still shutting down; ignoring.",
                signal.Signals(signum).name,
            )
            return
        # the two branches below return after `os._exit` only so that a caller
        # who stubbed it out -- which the tests do, having no use for a dead
        # interpreter -- cannot fall through and start the teardown over
        if phase is _Phase.CALLBACKS:
            # the peers are already out of the collective, so nothing is left to
            # protect. Someone escalating here -- a second Ctrl-C, or the
            # scheduler -- means it, even at the cost of the checkpoint.
            logger.info(
                "Received %s while running post-shutdown callbacks; the backend "
                "is already down, so exiting and abandoning the rest.",
                signal.Signals(signum).name,
            )
            os._exit(exit_code)
            return
        if phase is _Phase.COMPLETE:
            # the SystemExit from the first signal was swallowed (pytest turns
            # it into a test failure and keeps running; so does any bare
            # except), so being here means graceful exit failed. Honor the
            # convention that a repeated signal kills the process.
            logger.info(
                "Received %s after teardown already completed; exiting.",
                signal.Signals(signum).name,
            )
            os._exit(exit_code)
            return
        pending = _pending_stop
        if pending is not None:
            # Somebody is holding a rendezvous open that every rank reaches, so
            # record the intent and let them leave it together rather than tearing
            # the backend down from wherever this signal happened to land.
            phase = _Phase.STOP_REQUESTED
            pending.request(signum)
            write_marker("signal-deferred", signal=signal.Signals(signum).name)
            logger.info(
                "Received %s, stopping at the next cooperative stopping point.",
                signal.Signals(signum).name,
            )
            return
        logger.info(
            "Received %s, shutting down the distributed backend before exiting.",
            signal.Signals(signum).name,
        )
        terminate(exit_code)

    global _terminate, _installed_pid, _pending_stop
    previous = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    previous_terminate = _terminate
    previous_installed_pid = _installed_pid
    _terminate = terminate
    _installed_pid = installed_pid
    for sig in TERMINATION_SIGNALS:
        signal.signal(sig, handle)
    try:
        yield
    finally:
        # restored rather than cleared, for the same reason the signal
        # dispositions above are: nesting is legal, and clearing on an inner exit
        # would leave an outer scope's deferral silently doing nothing
        _terminate = previous_terminate
        _installed_pid = previous_installed_pid
        # cleared rather than restored, because `defer_termination` owns its own
        # lifetime and reaching here with one registered is a bug, not a nesting
        _pending_stop = None
        for sig, previous_handler in previous.items():
            signal.signal(sig, previous_handler)
        clear_post_shutdown_callbacks()
