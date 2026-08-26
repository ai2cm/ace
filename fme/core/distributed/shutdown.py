"""Exit safely when a job is preempted, by aborting NCCL before anything else.

Schedulers preempt jobs by sending SIGTERM and escalating to SIGKILL once a
grace period expires. A rank that is killed while a peer's NCCL kernels still
have in-flight NVLink accesses into its memory faults the surviving ranks with
``CUDA error: Invalid access of peer GPU memory over nvlink or a hardware
error``. That raises SXid errors, so the GPUs need a reboot and the cluster
cordons the node. To avoid that outcome, no rank may exit until every rank has
stopped its own NCCL kernels.

Each rank handles this on its own, with no cross-rank coordination. The
abort must not meet a kernel that is actively exchanging data: aborting one
faulted the fabric in a real preemption (8xH100, mid-training-step: contained
NVLink peer-access errors, an unrecoverable NVSwitch SXid, node cordoned),
while aborting idle or *waiting* kernels was validated safe on H100 and B200.
So the training, validation, and inference loops offer every batch boundary as
a stopping point (`park_if_terminating`), and a rank that reaches one drains
its device work and skips the abort entirely: it holds no kernels for an abort
to stop and no host thread for it to release, while freeing its communicator
buffers would fault the waiting kernels of a peer still wedged in a collective
it never joined (observed contained on H100, the same error class as the
fabric fault). Only a rank that misses the park deadline aborts --
``ncclCommAbort`` kills its local kernels and unblocks the host thread waiting
on them -- and by missing the deadline that rank has shown it is wedged in a
collective a peer will never complete, where the kernels are waiting rather
than transferring.

Every rank, aborting or not, then holds its exit until no peer can still be
aborting (see `_wait_for_peer_aborts`), so its memory stays mapped until every
peer's kernels have stopped. That floor is the only cross-rank timing here,
and it is deliberately loose: ranks compare nothing, and each measures from
its *own* signal, which torchrun's agent delivers to the local workers one
after another -- 1.1s apart on a 2-GPU node in the field, and the gap grows
with the world size.

Nothing on this path may wait forever. A hang here is worse than a hard exit:
a rank wedged in the GPU driver cannot be killed at all, SIGKILL included, so
its container never stops and the scheduler records a plain failure instead of
a preemption -- which costs the run its automatic requeue. So the abort is
bounded (`_abort_local_communicators`), the stderr writes cannot block
(`write_stderr`), and a watchdog armed at the signal
(`_start_hard_deadline_watchdog`) exits with the signal's code at a fixed
deadline whatever else is stuck.

The signal is received on a *listener* -- a dedicated thread whose only job is
to wait for the signal -- because the main thread cannot be trusted to receive
it: a preempted rank's main thread is typically blocked inside a collective,
where an ordinary Python signal handler would never run. How the listener
observes the signal anyway is `SignalListener`'s concern
(`fme.core.distributed._signal_listener`); everything that *happens* on
termination is decided here, in `_abort_and_exit` and the steps it calls.

The instants before the listener is armed remain unprotected --
``init_process_group`` itself, inside ``Distributed.context()`` entry, most
notably.
"""

import contextlib
import os
import select
import signal
import threading
import time
import traceback
from collections.abc import Callable, Generator
from typing import NamedTuple

from fme.core.device import in_dataloader_worker

from ._signal_listener import SignalListener

TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)

# Allowance for a peer's abort to run, on top of the park deadline it starts
# at and the settle period covering signal skew: no rank may exit before
# signal + park_deadline + settle_period + grace_period (see
# `_wait_for_peer_aborts`). Aborts return in under a second on a wedged 8-GPU
# node (measured on H100 and B200), so this is a generous cover, and the sum
# stays well inside torchrun's 30s SIGTERM-to-SIGKILL budget (`PContext.close`'s
# default timeout, torch/distributed/elastic/agent/server/local_elastic_agent.py).
DEFAULT_GRACE_PERIOD = 5.0

# How long the listener waits for the main thread to stop touching training
# state (see _ExitCoordination.mark_training_state_frozen) before giving up on the
# post-abort callbacks. One batch of compute bounds the wait in the common
# case; a rank that parked has already frozen, so the wait is free there.
DEFAULT_STATE_FREEZE_TIMEOUT = 5.0

# How long the listener waits for the main thread to reach a loop boundary
# before concluding it is wedged and aborting without it -- long enough for
# the tail of a training batch or an inference window (sub-second to a few
# seconds). A rank that parks within it never aborts at all, so this is also
# how long a healthy rank waits before starting its restart checkpoint.
DEFAULT_PARK_DEADLINE = 5.0

# Rank-to-rank signal skew the exit floor tolerates, on top of the park
# deadline an aborting peer starts from. torchrun's agent signals its local
# workers one after another, so a peer's clock may lag this rank's by seconds
# (1.1s apart on a 2-GPU node in the field, more as the world grows). Skew
# past `settle_period + grace_period` minus the abort's own duration would let
# this rank exit -- unmapping its memory -- while a peer's kernels are still
# waiting on it.
DEFAULT_SETTLE_PERIOD = 3.0

# How long the abort may take before the listener stops waiting on it.
# `ncclCommAbort` returns in under a second on a wedged 8-GPU node; one that
# has not returned in this long is wedged in the driver and will not return at
# all, and waiting on it forfeits the restart checkpoint and the orderly exit
# for nothing (see `_abort_local_communicators`).
DEFAULT_ABORT_BUDGET = 3.0

# When the watchdog gives up on an orderly termination and exits anyway,
# measured from this rank's signal. Clamped up to the exit floor, and kept
# inside torchrun's 30s SIGTERM-to-SIGKILL budget with room for the skew
# between the agent's clock and this rank's signal -- being killed is the
# outcome the watchdog exists to avoid.
DEFAULT_HARD_DEADLINE = 24.0

# Longest a single message may hold up the exit path. stderr's reader may be
# gone (EPIPE) or, worse, alive and not draining: a full pipe blocks a plain
# write indefinitely, which is not a wait this path may take.
_STDERR_WRITE_TIMEOUT = 1.0

_post_abort_callbacks: list[Callable[[], None]] = []


def add_post_abort_callback(callback: Callable[[], None]) -> None:
    """Run ``callback`` on the listener thread before exit, past the abort.

    Callbacks run only once the main thread has parked at a loop boundary or
    unwound into the context's exit, where it blocks until the process ends --
    so their reads of training state cannot race it. A main thread that has
    done neither within ``state_freeze_timeout`` (still computing, or stuck
    somewhere that is not a collective) forfeits the callbacks rather than
    risk a torn snapshot. Best-effort in duration too: the hard deadline caps
    how long a callback may take (see ``_start_hard_deadline_watchdog``).
    Peers may have aborted their communicators by now, so callbacks must not
    use collectives, nor the logging module (see ``write_stderr``).
    """
    _post_abort_callbacks.append(callback)


def clear_post_abort_callbacks() -> None:
    _post_abort_callbacks.clear()


def write_stderr(message: str) -> None:
    """Write to stderr from the listener thread or a post-abort callback.

    ``os.write``, not the logger: a logging handler's lock may be held by a
    thread blocked writing to a stalled stderr, and nothing on the exit path
    may deadlock on it. Never raises and never blocks for long: stderr's
    reader may already be gone mid-preemption (EPIPE) or still holding a full
    pipe it has stopped draining, and losing a message must not lose the
    abort, the grace period, or the exit. A message larger than the pipe's
    remaining room can still block after the readiness check; the watchdog
    covers that (see `_start_hard_deadline_watchdog`).
    """
    try:
        if not select.select([], [2], [], _STDERR_WRITE_TIMEOUT)[1]:
            return
        os.write(2, message.encode())
    except OSError:
        pass


class _ExitCoordination:
    """The facts the main thread and the listener thread tell each other.

    Each starts false and flips true exactly once; the methods wrap the
    thread-safe flag primitive (``threading.Event``) in the fact it records.
    """

    def __init__(self) -> None:
        self._listener_owns_exit = threading.Event()
        self._training_state_frozen = threading.Event()
        self._abort_started = threading.Event()

    def mark_listener_owns_exit(self) -> None:
        """A termination signal arrived: from now on the listener thread ends
        the process, and the main thread must not exit on its own.
        """
        self._listener_owns_exit.set()

    def listener_owns_exit(self) -> bool:
        return self._listener_owns_exit.is_set()

    def mark_training_state_frozen(self) -> None:
        """The main thread is parked at a loop boundary or blocked in the
        context's exit: it will launch no further collectives and never touch
        training state again, so the abort cannot meet an actively
        transferring kernel and the post-abort callbacks may read state
        without racing a mutation.
        """
        self._training_state_frozen.set()

    def mark_abort_started(self) -> None:
        """The listener has begun aborting the communicators; a main thread
        released by that abort no longer needs to drain device work on its
        way into the context's exit.
        """
        self._abort_started.set()

    def abort_started(self) -> bool:
        return self._abort_started.is_set()

    def wait_until_training_state_frozen(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds for the freeze; False on timeout."""
        return self._training_state_frozen.wait(timeout)


class _ArmedContext(NamedTuple):
    """What `park_if_terminating` needs from the one active listener context."""

    coordination: _ExitCoordination
    drain: Callable[[], None]


# The active listener context's coordination and drain, so that
# `park_if_terminating` -- called from the training loops, far from the
# context -- can see whether a termination is pending. At most one context is
# active at a time (`Distributed.context()` is non-nestable), and a forked
# child clears it (`_clear_armed_context_in_child`): the inherited copy of the
# parent's pending termination must not park the child's threads.
_armed_context: _ArmedContext | None = None
_at_fork_registered = False


def _clear_armed_context_in_child() -> None:
    global _armed_context
    _armed_context = None


def park_if_terminating() -> None:
    """Give a pending termination a safe place to happen: a loop boundary.

    The loops call this once per batch or window, at a point where this rank
    has launched every collective it is going to. With no termination signal
    pending -- the overwhelmingly common case -- it returns immediately, at
    the cost of one flag read. Once one is pending, the calling thread first
    *drains* its device work -- the loop reaching a boundary is a host-side
    fact, while the last batch's collectives may still be running on the GPU
    (DDP enqueues its gradient all-reduce asynchronously and nothing in a
    default-config batch blocks the host on it) -- and then blocks here
    permanently. That freezes training state at the boundary with the
    device idle, which tells the listener this rank has no kernels to stop --
    so it skips the abort (`_abort_and_exit`) -- and lets the post-abort
    callbacks snapshot a boundary-consistent state. The listener's exit ends
    the process out from under the parked thread.

    A drain that hangs (a collective a peer will never complete) or raises (a
    sticky device error) leaves the freeze unset, so the listener takes the
    deadline path, which assumes nothing about this rank's kernels.

    Only the main thread of a listener-armed process parks; any other caller
    returns immediately. Reaching a loop boundary in a DataLoader worker says
    nothing about the parent's collectives, so workers never park.
    """
    armed = _armed_context
    if armed is None or not armed.coordination.listener_owns_exit():
        return
    if threading.current_thread() is not threading.main_thread():
        return
    if in_dataloader_worker():
        return
    write_stderr(
        "Main thread reached a loop boundary for termination; draining "
        "local device work.\n"
    )
    try:
        armed.drain()
    except BaseException:
        write_stderr(f"Draining local device work failed:\n{traceback.format_exc()}")
    else:
        write_stderr("Main thread parked at a loop boundary for termination.\n")
        armed.coordination.mark_training_state_frozen()
    threading.Event().wait()


def _wait_for_main_thread_to_park(
    coordination: _ExitCoordination,
    park_deadline: float,
) -> bool:
    """Wait out the park deadline. True if training state froze within it.

    A parked main thread has drained its device work (`park_if_terminating`
    synchronizes before freezing) and launches no further collectives; a main
    thread that unwound into the context's exit drains there likewise. Either
    way the rank holds no kernels for an abort to stop and no host thread for
    it to release, so the caller skips the abort rather than free buffers a
    peer wedged in a collective this rank never joined is still polling. A
    main thread that never freezes is wedged in such a collective itself (or
    in a drain of one), which is the case the abort exists for: its kernels
    are waiting rather than transferring, and simultaneous aborts of waiting
    kernels are the probe-validated shape.
    """
    if coordination.wait_until_training_state_frozen(park_deadline):
        return True
    write_stderr(
        f"Main thread has not parked {park_deadline}s after the signal; "
        "aborting without it.\n"
    )
    return False


def _abort_local_communicators(abort: Callable[[], None], budget: float) -> None:
    """Stop this rank's own NCCL kernels and release its wedged host threads.

    Bounded, and on its own thread. An abort that has not returned within the
    budget is wedged in the driver and will not return, while the listener
    still owes the peers their grace period and the scheduler an exit that
    reads as a preemption. The thread is left behind; the exit ends it.
    """

    def run() -> None:
        try:
            abort()
        except BaseException:
            write_stderr(f"Aborting communicators failed:\n{traceback.format_exc()}")

    aborter = threading.Thread(target=run, name="fme-abort", daemon=True)
    aborter.start()
    aborter.join(budget)
    if aborter.is_alive():
        write_stderr(
            f"Aborting communicators has not returned {budget}s after it "
            "started; continuing to the exit without it.\n"
        )


def _run_post_abort_callbacks(
    coordination: _ExitCoordination, state_freeze_timeout: float
) -> None:
    """Run the registered callbacks once training state is safe to read.

    A rank that parked at a loop boundary froze its state there, so this
    reaches the callbacks immediately -- and, having skipped the abort, with
    the whole budget before the hard deadline left to write in.

    On the abort path the abort releases only a main thread blocked in a
    collective; one between collectives keeps computing until the next
    collective raises, and the callbacks' reads of training state would race
    it. Wait for it to block in the context's exit, and forfeit the callbacks
    rather than record a torn snapshot if it never does.
    """
    if not _post_abort_callbacks:
        return
    if not coordination.wait_until_training_state_frozen(state_freeze_timeout):
        write_stderr(
            f"Main thread still running {state_freeze_timeout}s after the "
            "abort; skipping post-abort callbacks.\n"
        )
        return
    for callback in _post_abort_callbacks:
        try:
            callback()
        except BaseException:
            write_stderr(f"Post-abort callback failed:\n{traceback.format_exc()}")


def _wait_for_peer_aborts(
    signal_time: float,
    grace_period: float,
    park_deadline: float,
    settle_period: float,
) -> None:
    """Peers' aborts must finish, so their kernels stop touching this rank's
    memory, before this rank exits.

    A peer that aborts starts at its own park deadline, measured from its own
    signal, which may lag this rank's by the skew in torchrun's per-worker
    signalling. So every rank holds its exit until the latest instant a peer
    can still be aborting: the park deadline, plus the settle period for that
    skew, plus the grace period for the abort itself. A rank that parked never
    aborts and so has no abort of its own to wait out, but it must still wait
    out its peers'.
    """
    exit_time = signal_time + park_deadline + settle_period + grace_period
    time.sleep(max(0.0, exit_time - time.monotonic()))


def _start_hard_deadline_watchdog(
    signum: signal.Signals, signal_time: float, deadline: float
) -> None:
    """Exit ``deadline`` seconds after the signal, whatever else is stuck.

    Every other step is bounded, so reaching this means a driver that has
    stopped answering, a callback still writing, a stderr message that
    outlasted its readiness check, or a bug. The scheduler's SIGKILL is not a
    backstop for any of them: a thread wedged in the GPU driver is unkillable,
    and a container that cannot stop is recorded as a failure rather than a
    preemption -- which is what costs the run its requeue.
    """

    def wait_then_exit() -> None:
        time.sleep(max(0.0, signal_time + deadline - time.monotonic()))
        write_stderr(
            f"Termination has not finished {deadline}s after the signal; "
            f"exiting with code {128 + signum} anyway.\n"
        )
        os._exit(128 + signum)

    threading.Thread(
        target=wait_then_exit, name="fme-termination-watchdog", daemon=True
    ).start()


def _abort_and_exit(
    signum: signal.Signals,
    abort: Callable[[], None],
    coordination: _ExitCoordination,
    grace_period: float,
    state_freeze_timeout: float,
    park_deadline: float,
    settle_period: float,
    abort_budget: float,
    hard_deadline: float,
) -> None:
    """The termination policy. Runs on the listener thread."""
    signal_time = time.monotonic()
    coordination.mark_listener_owns_exit()
    exit_floor = park_deadline + settle_period + grace_period
    _start_hard_deadline_watchdog(signum, signal_time, max(hard_deadline, exit_floor))
    write_stderr(f"Received {signal.Signals(signum).name}; terminating this rank.\n")
    if _wait_for_main_thread_to_park(coordination, park_deadline):
        write_stderr(
            "Main thread parked with the device idle; this rank has nothing "
            "to abort.\n"
        )
    else:
        coordination.mark_abort_started()
        _abort_local_communicators(abort, abort_budget)
    _run_post_abort_callbacks(coordination, state_freeze_timeout)
    _wait_for_peer_aborts(signal_time, grace_period, park_deadline, settle_period)
    write_stderr(f"Exiting with code {128 + signum}.\n")
    os._exit(128 + signum)


@contextlib.contextmanager
def abort_and_exit_on_termination(
    abort: Callable[[], None],
    grace_period: float = DEFAULT_GRACE_PERIOD,
    state_freeze_timeout: float = DEFAULT_STATE_FREEZE_TIMEOUT,
    park_deadline: float = DEFAULT_PARK_DEADLINE,
    settle_period: float = DEFAULT_SETTLE_PERIOD,
    abort_budget: float = DEFAULT_ABORT_BUDGET,
    hard_deadline: float = DEFAULT_HARD_DEADLINE,
    drain: Callable[[], None] | None = None,
) -> Generator[None, None, None]:
    """Exit on SIGTERM or SIGINT, aborting the distributed backend first.

    Once a termination signal has arrived, the listener thread owns the
    process's exit. That matters because the abort *releases* a main thread
    that was blocked in a collective, which then typically raises out of the
    training loop: leaving this context blocks until the listener's exit, so
    the unwinding main thread cannot exit first -- ahead of the grace period,
    and with the wrong exit code. Repeated signals are ignored for the same
    reason: nothing short of SIGKILL may cut the grace period short. The
    guarantee only holds for a main thread that unwinds through this context;
    nothing on the way out may ``os._exit`` or ``SIG_DFL`` its way around it.

    Args:
        abort: Locally aborts the backend's communicators. Called on its
            own thread, only when the main thread failed to park -- so it runs
            while that thread is blocked in a collective, and must not require
            its cooperation.
        grace_period: Allowance for a peer's abort to run and for signal
            skew: no rank exits before ``park_deadline + settle_period +
            grace_period`` after the signal, so peers' aborts finish while
            this process's memory is still mapped (see
            ``_wait_for_peer_aborts``).
        state_freeze_timeout: Seconds the listener waits for the main thread
            to block in this context's exit before skipping the post-abort
            callbacks (see ``add_post_abort_callback``).
        park_deadline: Seconds after the signal the listener waits for the
            main thread to park at a loop boundary before concluding it is
            wedged and aborting without it (see ``park_if_terminating`` and
            ``_wait_for_main_thread_to_park``).
        settle_period: Seconds of rank-to-rank signal skew the exit floor
            tolerates, on top of the park deadline an aborting peer starts
            from (see ``_wait_for_peer_aborts``).
        abort_budget: Seconds the listener waits for ``abort`` to return
            before moving on to the callbacks and the exit (see
            ``_abort_local_communicators``).
        hard_deadline: Seconds after the signal at which the process exits
            regardless of what is still stuck, clamped up to the exit floor
            (see ``_start_hard_deadline_watchdog``).
        drain: Blocks until every kernel this rank has enqueued has completed
            (``torch.cuda.synchronize`` or equivalent). Called on the main
            thread before it freezes training state, so that a frozen rank --
            which skips the abort -- is known to hold no running kernels of
            its own. None means there is nothing asynchronous to drain.
    """
    global _armed_context, _at_fork_registered
    if drain is None:

        def drain() -> None:
            pass

    if threading.current_thread() is not threading.main_thread():
        # only the main thread may install handlers; a thread shares the
        # process disposition its main thread installed
        yield
        return
    if in_dataloader_worker():
        # a spawn- or forkserver-started worker enters this context only to
        # learn its rank, and the DataLoader owns its lifecycle: exiting out
        # from under it is not ours to do
        yield
        return

    coordination = _ExitCoordination()
    listener = SignalListener(
        signals=TERMINATION_SIGNALS,
        on_signal=lambda signum: _abort_and_exit(
            signum,
            abort,
            coordination,
            grace_period,
            state_freeze_timeout,
            park_deadline,
            settle_period,
            abort_budget,
            hard_deadline,
        ),
    )
    if not _at_fork_registered:
        os.register_at_fork(after_in_child=_clear_armed_context_in_child)
        _at_fork_registered = True
    _armed_context = _ArmedContext(coordination, drain)
    listener.start()
    try:
        yield
    finally:
        # Past this point the main thread no longer touches training state, so
        # this block may mark it frozen -- but a freeze tells the listener it
        # is safe to abort, so device work must drain first. The drain runs
        # unconditionally rather than only when a termination is pending,
        # because the listener may not have read a just-delivered signal yet
        # when an exception unwinds here alongside it. On a normal exit the
        # device is idle, so the drain is instant; a drain wedged on a dead
        # peer's collective is released by the deadline abort once the
        # scheduler's (or torchrun's teardown) SIGTERM arrives. Once the abort
        # has started, draining would only raise against the torn-down
        # communicators and forfeit the restart checkpoint, so it is skipped.
        if not coordination.abort_started():
            try:
                drain()
            except BaseException:
                write_stderr(
                    f"Draining local device work failed:\n{traceback.format_exc()}"
                )
            else:
                coordination.mark_training_state_frozen()
        else:
            coordination.mark_training_state_frozen()
        listener.request_stop()
        if not coordination.listener_owns_exit():
            # covers the race where a signal was delivered but not yet read:
            # long enough for the listener to abort and exit
            listener.wait_until_finished(
                timeout=park_deadline + settle_period + grace_period + 10.0
            )
        if coordination.listener_owns_exit():
            # hold the unwinding main thread for the abort and grace period,
            # with the listener still armed -- dismantling it here would
            # restore the previous dispositions (typically SIG_DFL) and let a
            # repeated signal kill the rank before the grace period ends
            listener.block_until_process_exit()
        else:
            listener.dismantle()
            _armed_context = None
            # the callbacks belonged to this context's session; a later
            # context must not fire them against torn-down state
            clear_post_abort_callbacks()
