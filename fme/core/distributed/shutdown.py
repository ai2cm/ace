"""Exit safely when a job is preempted, by aborting NCCL before anything else.

Schedulers preempt jobs by sending SIGTERM and escalating to SIGKILL once a
grace period expires. A rank that is killed while a peer's NCCL kernels still
have in-flight NVLink accesses into its memory faults the surviving ranks with
``CUDA error: Invalid access of peer GPU memory over nvlink or a hardware
error``. That raises SXid errors, so the GPUs need a reboot and the cluster
cordons the node. To avoid that outcome, no rank may exit until every rank has
stopped its own NCCL kernels.

Each rank handles this on its own, with no cross-rank coordination: on a
termination signal it aborts its own communicators -- ``ncclCommAbort`` kills
the local kernels and unblocks whatever host thread was waiting on them --
then waits out a grace period so its peers' aborts finish before it exits.

The abort itself is deferred until this rank's collectives are idle or merely
waiting. Aborting communicators whose kernels are actively exchanging data
faulted the fabric in a real preemption (8xH100, mid-training-step: contained
NVLink peer-access errors, an unrecoverable NVSwitch SXid, node cordoned),
while aborting idle or *waiting* kernels was validated safe on H100 and B200.
So the training, validation, and inference loops offer every batch boundary
as a stopping point (`park_if_terminating`), and the listener aborts only
once the main thread has parked at one -- or has unwound into the context's
exit -- plus a settle period for peers' kernel tails. A main thread that
misses the deadline is wedged in a collective a peer will never complete,
where the kernels are waiting rather than transferring and the abort is safe
anyway.

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
import signal
import threading
import time
import traceback
from collections.abc import Callable, Generator
from typing import NamedTuple

from fme.core.device import in_dataloader_worker

from ._signal_listener import SignalListener

TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)

# How long an aborting rank waits for its peers' aborts before exiting. Aborts
# return in under a second on a wedged 8-GPU node (measured on H100 and B200),
# so this generously covers rank-to-rank signal skew while fitting many times
# over in torchrun's shared 30s SIGTERM-to-SIGKILL budget (`PContext.close`'s
# default timeout, torch/distributed/elastic/agent/server/local_elastic_agent.py).
DEFAULT_GRACE_PERIOD = 5.0

# How long the listener waits for the main thread to stop touching training
# state (see _ExitCoordination.mark_training_state_frozen) before giving up on the
# post-abort callbacks. One batch of compute bounds the wait in the common
# case; together with the callbacks and grace period this must fit torchrun's
# 30s SIGTERM-to-SIGKILL budget.
DEFAULT_STATE_FREEZE_TIMEOUT = 5.0

# How long the listener waits for the main thread to park at a loop boundary
# before aborting anyway. Long enough for the tail of a training batch or an
# inference window (sub-second to a few seconds); a main thread that misses it
# is taken to be wedged in a collective a peer will never complete, whose
# kernels are waiting rather than transferring, so the abort is safe without
# the park. With the settle period, abort, post-abort callbacks, and grace
# period behind it, the sum must fit the same 30s budget.
DEFAULT_PARK_DEADLINE = 10.0

# How long a parked rank waits before aborting. Parking means this rank's own
# kernels have drained (`park_if_terminating` synchronizes the device before
# freezing), but a peer's kernel for a collective this rank has finished may
# still be retiring -- actively reading this rank's memory for a little
# longer. Kernel tails are sub-millisecond and signal skew across ranks is
# tens of milliseconds, so this is a generous cover.
DEFAULT_SETTLE_PERIOD = 1.0

_post_abort_callbacks: list[Callable[[], None]] = []


def add_post_abort_callback(callback: Callable[[], None]) -> None:
    """Run ``callback`` on the listener thread after the abort, before exit.

    Callbacks run only once the main thread has unwound into the context's
    exit, where it blocks until the process ends -- so their reads of training
    state cannot race it. A main thread that has not unwound within
    ``state_freeze_timeout`` (still computing, or stuck somewhere that is not
    a collective) forfeits the callbacks rather than risk a torn snapshot.
    Best-effort in duration too: the scheduler's SIGKILL caps how long a
    callback may take. The communicators are gone by then, so callbacks must
    not use collectives, nor the logging module (see ``write_stderr``).
    """
    _post_abort_callbacks.append(callback)


def clear_post_abort_callbacks() -> None:
    _post_abort_callbacks.clear()


def write_stderr(message: str) -> None:
    """Write to stderr from the listener thread or a post-abort callback.

    ``os.write``, not the logger: a logging handler's lock may be held by a
    thread blocked writing to a stalled stderr, and nothing on the exit path
    may deadlock on it. Never raises: stderr's reader may already be gone
    mid-preemption (EPIPE), and losing a message must not lose the abort,
    the grace period, or the exit.
    """
    try:
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
    device idle, which tells the listener it may abort
    (`_wait_for_main_thread_to_park`) and lets the post-abort callbacks
    snapshot a boundary-consistent state. The listener's exit ends the
    process out from under the parked thread; the scheduler's SIGKILL is the
    backstop.

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
    coordination: _ExitCoordination, park_deadline: float, settle_period: float
) -> None:
    """Hold the abort until this rank's collectives are idle or merely waiting.

    A parked main thread has drained its device work (`park_if_terminating`
    synchronizes before freezing) and launches no further collectives -- so
    after a settle period for peers' kernel tails (see
    ``DEFAULT_SETTLE_PERIOD``), nothing is actively moving data and the abort
    is safe. A main thread that unwound into the context's exit before the
    abort drains there likewise. A main thread that never freezes is wedged
    in a collective a peer will never complete (or in a drain of one); its
    kernels are waiting, not transferring, which is also safe to abort -- so
    after the deadline the abort proceeds without the park.
    """
    if coordination.wait_until_training_state_frozen(park_deadline):
        time.sleep(settle_period)
    else:
        write_stderr(
            f"Main thread has not parked {park_deadline}s after the signal; "
            "aborting without it.\n"
        )


def _abort_local_communicators(abort: Callable[[], None]) -> None:
    """Stop this rank's own NCCL kernels and release its wedged host threads."""
    try:
        # not bounded: if the abort hangs, exiting anyway would guarantee the
        # peer-GPU fault, so the rank rides to the scheduler's SIGKILL instead
        abort()
    except BaseException:
        write_stderr(f"Aborting communicators failed:\n{traceback.format_exc()}")


def _run_post_abort_callbacks(
    coordination: _ExitCoordination, state_freeze_timeout: float
) -> None:
    """Run the registered callbacks once training state is safe to read.

    The abort releases only a main thread blocked in a collective; one between
    collectives keeps computing until the next collective raises, and the
    callbacks' reads of training state would race it. Wait for it to block in
    the context's exit, and forfeit the callbacks rather than record a torn
    snapshot if it never does.
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


def _wait_for_peer_aborts(grace_period: float) -> None:
    """Peers' aborts must finish, so their kernels stop touching this rank's
    memory, before this rank exits.
    """
    time.sleep(grace_period)


def _abort_and_exit(
    signum: signal.Signals,
    abort: Callable[[], None],
    coordination: _ExitCoordination,
    grace_period: float,
    state_freeze_timeout: float,
    park_deadline: float,
    settle_period: float,
) -> None:
    """The termination policy. Runs on the listener thread."""
    coordination.mark_listener_owns_exit()
    write_stderr(
        f"Received {signal.Signals(signum).name}, aborting distributed "
        "communicators before exiting.\n"
    )
    _wait_for_main_thread_to_park(coordination, park_deadline, settle_period)
    coordination.mark_abort_started()
    _abort_local_communicators(abort)
    _run_post_abort_callbacks(coordination, state_freeze_timeout)
    _wait_for_peer_aborts(grace_period)
    write_stderr(f"Exiting with code {128 + signum}.\n")
    os._exit(128 + signum)


@contextlib.contextmanager
def abort_and_exit_on_termination(
    abort: Callable[[], None],
    grace_period: float = DEFAULT_GRACE_PERIOD,
    state_freeze_timeout: float = DEFAULT_STATE_FREEZE_TIMEOUT,
    park_deadline: float = DEFAULT_PARK_DEADLINE,
    settle_period: float = DEFAULT_SETTLE_PERIOD,
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
        abort: Locally aborts the backend's communicators. Called on the
            listener thread, typically while the main thread is blocked in a
            collective or parked at a loop boundary, so it must not require
            the main thread's cooperation.
        grace_period: Seconds to wait between aborting and exiting, so that
            peers' aborts finish while this process's memory is still mapped.
        state_freeze_timeout: Seconds the listener waits for the main thread
            to block in this context's exit before skipping the post-abort
            callbacks (see ``add_post_abort_callback``).
        park_deadline: Seconds the listener waits for the main thread to park
            at a loop boundary before aborting anyway (see
            ``park_if_terminating`` and ``_wait_for_main_thread_to_park``).
        settle_period: Seconds a parked rank waits before aborting, covering
            peers' kernel tails.
        drain: Blocks until every kernel this rank has enqueued has completed
            (``torch.cuda.synchronize`` or equivalent). Called on the main
            thread before it freezes training state, so a settle-path abort
            cannot meet this rank's own kernels still running. None means
            there is nothing asynchronous to drain.
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
        # the main thread only winds the listener down from here; it will not
        # touch training state again, so the post-abort callbacks may read it.
        # Every freeze is preceded by a drain unless the abort already fired:
        # gating the drain on a pending termination instead would race the
        # listener thread, which may not have read the signal yet when an
        # exception unwinds here alongside it -- and a freeze set without a
        # drain lets the settle-path abort meet active kernels. On a normal
        # exit the device is idle and the drain is instant; a drain wedged on
        # a dead peer's collective is released by the deadline abort once the
        # scheduler's (or torchrun's teardown) SIGTERM arrives. After the
        # abort, draining would only raise against torn-down communicators
        # and forfeit the wedged path's restart checkpoint.
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
            listener.wait_until_finished(timeout=grace_period + 10.0)
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
