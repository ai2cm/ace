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
# state (see _ExitCoordination.training_state_frozen) before giving up on the
# post-abort callbacks. One batch of compute bounds the wait in the common
# case; together with the callbacks and grace period this must fit torchrun's
# 30s SIGTERM-to-SIGKILL budget.
DEFAULT_STATE_FREEZE_TIMEOUT = 5.0

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
    """The two facts the main thread and the listener thread tell each other.

    Each starts false and flips true exactly once; the methods wrap the
    thread-safe flag primitive (``threading.Event``) in the fact it records.
    """

    def __init__(self) -> None:
        self._listener_owns_exit = threading.Event()
        self._training_state_frozen = threading.Event()

    def mark_listener_owns_exit(self) -> None:
        """A termination signal arrived: from now on the listener thread ends
        the process, and the main thread must not exit on its own.
        """
        self._listener_owns_exit.set()

    def listener_owns_exit(self) -> bool:
        return self._listener_owns_exit.is_set()

    def mark_training_state_frozen(self) -> None:
        """The main thread will never touch training state again, so the
        post-abort callbacks may read it without racing a mutation.
        """
        self._training_state_frozen.set()

    def wait_until_training_state_frozen(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds for the freeze; False on timeout."""
        return self._training_state_frozen.wait(timeout)


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
) -> None:
    """The termination policy. Runs on the listener thread."""
    coordination.mark_listener_owns_exit()
    write_stderr(
        f"Received {signal.Signals(signum).name}, aborting distributed "
        "communicators before exiting.\n"
    )
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
            collective, so it must not require the main thread's cooperation.
        grace_period: Seconds to wait between aborting and exiting, so that
            peers' aborts finish while this process's memory is still mapped.
        state_freeze_timeout: Seconds the listener waits for the main thread
            to block in this context's exit before skipping the post-abort
            callbacks (see ``add_post_abort_callback``).
    """
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
            signum, abort, coordination, grace_period, state_freeze_timeout
        ),
    )
    listener.start()
    try:
        yield
    finally:
        # the main thread only winds the listener down from here; it will not
        # touch training state again, so the post-abort callbacks may read it
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
            # the callbacks belonged to this context's session; a later
            # context must not fire them against torn-down state
            clear_post_abort_callbacks()
