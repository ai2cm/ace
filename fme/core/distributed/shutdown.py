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
**Only a scope that opens a deferral gets that**, and today that is one training
loop -- `fme.core.generics.trainer.Trainer.train_one_epoch`, and not
`fme/downscaling/train.py`'s `Trainer.train_one_epoch`, which has a batch loop of
its own and does not open a deferral. A signal arriving during validation, inline
inference, or LR tuning is still acted on where it lands, and those paths hold
collectives of their own. The deferral is the mechanism, not a claim about
coverage.

So this module owns three things: *deferral* -- ``defer_termination`` makes the
handler record intent rather than act, and performs the teardown when the scope
it guards is left -- the evidence marker writer ``write_marker``, whose lines
are the only per-rank record that survives non-root ranks' log level, and the
watchdog, which now aborts the local communicator before its hard exit.
Deciding *when* a deferred stop is acted on is the caller's, not this module's:
nothing here knows about process groups or collectives, which is what keeps the
signal handling testable single-rank and in process.

The ``fme-stop:`` event vocabulary
----------------------------------

**This is the canonical list of every event any module writes**, kept here because
this module owns `write_marker`; the other modules point at it rather than
restating it, and adding an event means adding an entry. Each entry gives the
writing module, what the event means, and what a reader does with it. Every line
carries ``rank``, ``local_rank``, ``pid``, ``installed_pid``, ``wall`` and
``mono`` first, in that order, before the fields named below.

``signal-deferred`` -- this module, `handle_termination_signals`
    A termination signal was recorded rather than acted on. Its presence on *any*
    rank is what separates a preemption from a hang.
``signal-ignored`` -- this module, `handle_termination_signals`
    A further signal arrived while a stop was already pending. Expected on every
    real preemption: the scheduler signals the container, then torchrun's agent
    signals every rank again.
``shutdown-returned`` -- this module, ``terminate``
    This rank's collective teardown returned, in ``elapsed``, rather than riding
    to the scheduler's SIGKILL inside it.
``hard-exit`` -- this module, ``terminate``
    The process is ending in ``os._exit`` with ``code``. That code is what a
    reader has instead of an exit status, no ``SystemExit`` being raised here.
``deferral-overrun`` -- this module, `_warn_after`
    A stop stayed pending for twice the budget without reaching a stopping point.
    This rank may be SIGKILLed with its communicators open, so the node's GPUs may
    need resetting.
``watchdog-abort`` -- this module, `_hard_exit_after`
    The collective teardown overran ``timeout`` and the local communicator is
    being aborted, so this rank's teardown did not return.
``watchdog-abort-unavailable`` -- `fme.core.distributed.torch_distributed`
    The installed torch has no ``ProcessGroup.abort()``, so the watchdog fell
    straight through to its hard exit. It distinguishes that from an abort that
    was attempted and failed, which the watchdog would otherwise swallow
    identically.
``agreement-bound`` -- `fme.core.distributed.cooperative_stop`
    Which deadline this rank applied, in ``bound``. ``budget`` is the design's
    short bound; ``group-timeout`` means it was unavailable on this torch release,
    so a give-up waits the default group's own timeout -- see
    `fme.core.distributed.stop_agreement.timeout_contract_verified`.
``agreement-expired`` -- `fme.core.distributed.cooperative_stop`
    This rank reached an exchange with its budget already spent, so the floor
    applied instead. It arrived late, normally out of a checkpoint write.
``stop-agreed`` -- `fme.core.distributed.cooperative_stop`
    One exchange completed, carrying ``batch``, ``world`` and ``reason``. **The
    reader recipe:** count these lines against ``world=`` -- the rank with no line
    is the one that never reached the boundary.
``index-mismatch`` -- `fme.core.distributed.cooperative_stop`
    Ranks contributed different iteration indices, ``min`` and ``max``. Diagnostic
    only, written once per loop: the loops have desynchronised.
``agreement-timeout`` -- `fme.core.distributed.cooperative_stop`
    This rank gave up waiting for a peer that never reached ``batch``. Whether
    that is a preemption this rank's own signal did not reach or a genuine hang is
    told by ``signal-deferred`` appearing anywhere in the job's log.
``agreement-abandoned`` -- `fme.core.distributed.cooperative_stop`
    The gloo agreement group is held for the life of the process rather than
    reclaimed. This is the only record of it, and the reason the rank then exits
    hard.
"""

import contextlib
import enum
import logging
import os
import signal
import sys
import threading
import time
import traceback
import types
from collections.abc import Callable, Generator
from typing import Protocol

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
# rather than 30s each, so a slow rank spends its peers' allowance too.
#
# 20s of that 30s is this. What the agreement ahead of it spends is not a
# constant: with `DEFAULT_STOP_AGREEMENT_BUDGET` = 5s and a floor of
# `_MIN_DEADLINE` = 3s under every deadline, a rank whose local event was T
# seconds before the boundary it next reaches finishes agreeing by max(5, T + 3),
# so the callbacks -- the restart checkpoint -- begin with min(5, 7 - T) seconds
# of the window left and with none at all once T >= 7s. The arithmetic is worked
# out where those two constants are defined.
#
# There is no margin beyond that: this is a ceiling for a teardown that normally
# returns in well under a second, not a slice reserved for it, but a teardown that
# really does use its 20s leaves the checkpoint a sixth of the window at best.
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


class _Terminate(Protocol):
    """The teardown `handle_termination_signals` publishes.

    A `Protocol` rather than `Callable[..., None]`, which types the positional
    argument and nothing else: the dispatch in `defer_termination` passes
    ``exit_process=`` and ``hard=``, and under the looser annotation a misspelled
    or wrongly-typed keyword there would reach production unchecked.
    """

    def __call__(
        self, exit_code: int, *, exit_process: bool = True, hard: bool = False
    ) -> None: ...


# Published by `handle_termination_signals` so that a deferral, which has no
# access to that function's locals, can run the same teardown the handler would
# have run. `None` means no handler is installed and there is nothing to tear
# down -- the state every pytest session is in.
_terminate: _Terminate | None = None

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


# Longest field value a marker line will carry. gloo's messages run to a few
# hundred characters, and a line has to stay far under `PIPE_BUF` for the atomicity
# `write_marker` describes.
_MAX_FIELD_LENGTH: int = 160


def _field_value(value: str) -> str:
    """Make one value fit the ``key=value`` contract, whatever it contains.

    The contract is fixed space-separated ``key=value`` pairs, so a value may
    contain neither a space nor a newline: a torch error message contains both,
    and passing one through unencoded makes the line unparseable by the very
    readers these lines exist for -- including the tests' own field parser. Runs of
    whitespace become a single ``_`` rather than being percent-encoded, because a
    human reads these lines first.
    """
    collapsed = "_".join(value.split())
    if len(collapsed) <= _MAX_FIELD_LENGTH:
        return collapsed
    return collapsed[:_MAX_FIELD_LENGTH] + "..."


def write_marker(event: str, **fields: str) -> None:
    """Write one machine-readable evidence line to stderr. Never raises.

    ``logging`` cannot carry this: `fme/core/logging_utils.py` puts every
    non-root rank at ERROR, so an INFO line from ranks 1-7 never reaches the
    container log -- and the two things a reader needs are per-rank and negative
    (which rank failed to reach the stopping point, which rank's teardown did not
    return), so a rank-0 summary cannot serve. ``os.write`` also cannot deadlock
    on the logging lock, which matters because these lines are emitted from
    signal handlers and timer threads.

    One line is one ``write``, which is what keeps several ranks' lines from
    interleaving -- but only where that guarantee holds: a write under ``PIPE_BUF``
    (4096 on Linux) is atomic *on a pipe*, while a container that redirects stderr
    to a regular file relies on ``O_APPEND`` writes not interleaving instead, which
    is a different promise. Lines here stay far under 4096 bytes, so the stronger
    case applies wherever it applies at all.

    ``installed_pid`` comes from the module-level `_installed_pid` rather than
    from a parameter, so that callers with no access to the handler's locals --
    timer threads, and the loop-facing layer above this module -- emit lines one
    parser stays valid for. It is ``?`` where no handler is installed.

    Every field value goes through `_field_value`, so the ``key=value`` contract
    holds for all of them by construction rather than by each caller remembering
    it. A caller passing a torch error message is the case that needs it.
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
            *(f"{name}={_field_value(value)}" for name, value in fields.items()),
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
        # the caller has left something behind that interpreter finalization
        # cannot dispose of in bounded time; see `require_hard_exit`
        self.hard_exit = False
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

    def require_hard_exit(self) -> None:
        """Ask for `os._exit` rather than `sys.exit` once the teardown is done.

        For a caller that has left something behind which interpreter finalization
        cannot dispose of in bounded time.

        **This is the canonical statement of the measurement the whole hard-exit
        rule rests on**; every other module and test that needs it refers here
        rather than restating the figures, so that they cannot drift apart. The case
        it exists for is an abandoned gloo group: ``~ProcessGroupGloo()`` joins the
        worker thread still holding the abandoned operation, that destructor runs
        during ``Py_Finalize``, and measured on torch 2.7.1 the wait ends only when
        the wedged peer's socket closes -- 119.3s against a peer parked for 120s,
        24.3s against one parked for 25s, versus 0.05s with `os._exit`. Whether
        finalization joins that thread or leaks it is not deterministic; the same
        shape measured through a module global, with only the default group
        destroyed, leaked it and died in 4.66s under `sys.exit` against 3.02s under
        `os._exit`. So `sys.exit` can hand the rank to torchrun's SIGKILL, and the
        launcher then reads a signal death instead of the ``128 + signum`` this
        module promises.

        This does not weaken the rule that a hard exit is the fault to avoid. The
        watchdog's `os._exit` is dangerous because it can drop live communicators
        and skip the callbacks; this one runs *after* `shutdown` destroyed them and
        *after* the callbacks completed, so nothing is skipped and no communicator
        is dropped -- there is nothing left for finalization to do but block.
        """
        self.hard_exit = True

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


def _system_exit_code(code: object) -> int:
    """The integer ``os._exit`` needs, from a ``SystemExit.code`` that may be anything.

    ``SystemExit.code`` is ``int | str | None`` by CPython's own contract -- the
    isinstance narrowing here is of a stdlib union this repository does not own, so
    there is no type to refactor. ``None`` is a clean exit and a string is a message
    the interpreter would have printed before exiting 1, which is what the hard exit
    reports instead.
    """
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return _EXCEPTION_EXIT_CODE


def _print_pending_traceback() -> None:
    """Print the propagating exception to stderr, because nothing else will.

    Reached only where the dispatch below is about to hard-exit while an exception
    is unwinding: ``os._exit`` runs inside the ``finally`` that is unwinding it, so
    the exception never reaches the interpreter's own handler and its traceback
    would be lost. Keeping that traceback is the only reason the ordinary exception
    path declines to exit at all, so the hard path prints it here rather than
    trading it away.
    """
    try:
        traceback.print_exc()
        sys.stderr.flush()
    except BaseException:
        # evidence about a failure may not become a second failure during one
        pass


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
        raise RuntimeError(
            "Nested defer_termination() is not supported: the registry is "
            "process-global and only one scope can own the teardown. "
            "`fme.core.distributed.cooperative_stop` opens one, so two "
            "cooperative_stop scopes at once -- nested batch loops, most likely -- "
            "reach here as well."
        )
    pending = PendingStop(budget)
    _pending_stop = pending
    raising = False
    exiting = False
    # only meaningful once `exiting`; the caller's own code, so that a hard exit
    # reports what its `sys.exit` asked for rather than a substitute
    caller_exit_code = _EXCEPTION_EXIT_CODE
    try:
        yield pending
    except SystemExit as err:
        # Not `raising`. A caller leaving by `sys.exit` is asking for the very exit
        # the dispatch below performs rather than surfacing a crash, and there is no
        # traceback to protect -- so it takes the exit path, which is the only one
        # that can honour `require_hard_exit`. `cooperative_stop` leaves this way
        # when the loop-entry exchange is given up on, since the loop body must not
        # run against peers that are gone.
        exiting = True
        caller_exit_code = _system_exit_code(err.code)
        raise
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
        elif raising:
            # A propagating exception wins, and it wins even over a recorded
            # signal. `_terminate` raises `SystemExit` from this `finally`, so
            # exiting here would *replace* the exception -- and the exception this
            # discards is exactly the one the agreement went out of its way to
            # surface: a peer's `Connection closed by peer`, re-raised at the
            # boundary rather than masked as a graceful stop, on a rank that
            # (because the scheduler signals the whole group) has usually recorded
            # a signal too. So tear down, run the callbacks, and return, leaving
            # the traceback to reach the interpreter.
            #
            # What that trades away is the *process's* exit code: a rank that was
            # preempted and also raised reports 1, and the preempted-versus-failed
            # distinction is precisely what `128 + signum` carries to the scheduler.
            # The code passed here still follows the signal, but from this branch it
            # reaches only the watchdog's hard-exit backstop, so it does not restore
            # that distinction where anything outside the process can read it.
            # Masking a crash as a graceful stop would hide the crash, which is the
            # worse of the two. `CooperativeStop._local_reason` gives a recorded
            # signal the opposite precedence, for the reason set out there, and the
            # two are consistent: the *reason* peers read is the signal, because the
            # scheduler signalled the whole group, while the *exit code* this rank
            # reports is the exception's, because its traceback is the new
            # information.
            if pending.hard_exit:
                # Returning here would leave the rank in interpreter finalization
                # until the wedged peer dies, and torchrun would SIGKILL it inside
                # the 30s window -- so the launcher would read a signal death, which
                # is the very loss the paragraph above calls unacceptable in the
                # other direction. It would also hold this rank's GPU allocation for
                # that whole window with its backend already destroyed. The
                # traceback, which is the only thing `exit_process=False` was
                # protecting, is printed here instead; the exit code is the
                # exception's, which is what the propagating exception would have
                # produced anyway.
                #
                # A hard exit is safe *here specifically*, where the watchdog's is
                # not: `_terminate` takes it only after `shutdown` destroyed the
                # communicators and after the callbacks ran, so no communicator is
                # dropped and no work is skipped. There is nothing left for
                # finalization to do but block.
                _print_pending_traceback()
                _terminate(_EXCEPTION_EXIT_CODE, hard=True)
            else:
                _terminate(
                    pending.exit_code if pending.requested else _EXCEPTION_EXIT_CODE,
                    exit_process=False,
                )
        elif pending.requested:
            # A recorded signal takes precedence over a peer's deliberately: a
            # preemption is not a failure, and `128 + signum` is what the scheduler
            # and torchrun expect from a rank asked to stop.
            _terminate(pending.exit_code, hard=pending.hard_exit)
        elif pending.peer_stop:
            _terminate(_PEER_STOP_EXIT_CODE, hard=pending.hard_exit)
        elif exiting:
            # A `sys.exit` of the caller's own, with no stop recorded anywhere: tear
            # the backend down and run the callbacks, then let the caller's own code
            # reach the interpreter, exactly as the exception path does -- unless a
            # hard exit was asked for, in which case returning would leave the rank
            # in finalization until a wedged peer dies. The caller's own exit code is
            # carried through, since that is what its `sys.exit` asked for.
            if pending.hard_exit:
                _terminate(caller_exit_code, hard=True)
            else:
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

    An abort that is unavailable rather than merely unsuccessful is reported
    separately, because the ``except BaseException`` below cannot tell the two
    apart: see ``watchdog-abort-unavailable`` in the event vocabulary and
    `fme.core.distributed.torch_distributed`, which owns the check.
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

    def terminate(
        exit_code: int, *, exit_process: bool = True, hard: bool = False
    ) -> None:
        """Release the backend, run the callbacks, and exit.

        The only teardown implementation there is, reached either from the
        handler or from a deferral being left, so the phase transitions are
        identical whichever entry ran.

        `exit_process=False` is for a caller that is already propagating an
        exception: everything happens except the final `sys.exit`, so the
        exception goes on propagating with its traceback intact.

        `hard` replaces that final `sys.exit` with `os._exit`, for a caller that
        has left something behind which interpreter finalization cannot dispose of
        in bounded time -- see `PendingStop.require_hard_exit`, which is the only
        thing that sets it. It is reached only here, after `shutdown` and after the
        callbacks, so unlike the watchdog's hard exit it drops no communicator and
        skips no work.
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
        # Trial-only: separates "the restart write outlasted the launcher's grace
        # window" from "interpreter finalization blocked afterwards". Absent means
        # a callback had not returned when the process died.
        write_marker(
            "restart-write-returned", elapsed=f"{time.monotonic() - started:.2f}s"
        )
        if not exit_process:
            return
        if hard:
            write_marker("hard-exit", code=str(exit_code))
            # `os._exit` skips the buffered streams, and the callbacks above are
            # the code most likely to have written to them
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except BaseException:
                    pass
            os._exit(exit_code)
        # reached in production only when `hard` is false; a test that stubbed
        # `os._exit` out falls through to here rather than carrying on inside the
        # scope that asked to be left, which is the same guard the handler's own
        # hard-exit branches carry
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
    previous_callbacks = list(_post_shutdown_callbacks)
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
        # restored rather than cleared, for the same reason as the two globals
        # above: this context claims nesting is legal, and an unconditional clear
        # would take an outer context's callbacks with it when an inner one exits.
        # Callbacks registered *inside* this scope are still discarded, which is
        # what the clear was for -- the registry is process-global and its entries
        # close over objects this scope's job built.
        _post_shutdown_callbacks[:] = previous_callbacks
