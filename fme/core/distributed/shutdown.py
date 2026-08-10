"""Quiesce the GPU fabric before a preempted job exits.

Schedulers preempt jobs by sending SIGTERM and escalating to SIGKILL once a
grace period expires. A rank that dies while a peer's NCCL kernels still have
in-flight NVLink accesses into its memory faults the fabric (SXid errors), so
the GPUs need a reset and the cluster cordons the node. The invariant is
therefore that no rank exits until every rank has stopped its own NCCL
kernels.

Each rank quiesces itself, with no cross-rank coordination: on a termination
signal it aborts its own communicators -- ``ncclCommAbort`` kills the local
kernels and unblocks whatever host thread was waiting on them -- then waits
out a grace period so its peers' aborts finish before it exits.

A Python signal handler runs only when the main thread returns to the
interpreter, and a preempted rank's main thread is typically blocked inside a
collective's stream sync, so a handler could never run in time. Instead the
signal is observed on a dedicated listener thread through
``signal.set_wakeup_fd``, which CPython's C-level handler writes regardless of
what the main thread is doing.
"""

import contextlib
import os
import signal
import threading
import time
import traceback
import types
from collections.abc import Callable, Generator

from fme.core.device import in_dataloader_worker

TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)

# How long an aborting rank waits for its peers' aborts before exiting. Aborts
# return in under a second on a wedged 8-GPU node (measured on H100 and B200),
# so this generously covers rank-to-rank signal skew while fitting many times
# over in torchrun's shared 30s SIGTERM-to-SIGKILL budget (`PContext.close`'s
# default timeout, torch/distributed/elastic/agent/server/local_elastic_agent.py).
DEFAULT_GRACE_PERIOD = 5.0

# tells the sentinel apart from signal numbers, which are all positive
_STOP_LISTENING = 0

_armed = False  # whether the wakeup fd currently belongs to this module
_wakeup_write_fd: int | None = None
_at_fork_registered = False


def _disarm_in_child() -> None:
    """Undo the parent's signal setup in a forked child.

    A fork-started DataLoader worker inherits the ignore-handler and the wakeup
    fd, which still feeds the parent's pipe: a signal delivered to the worker
    would masquerade as the parent's own preemption. Restore the defaults so
    the child dies as it would have without the inheritance.
    """
    if not _armed:
        return
    try:
        for sig in TERMINATION_SIGNALS:
            signal.signal(sig, signal.SIG_DFL)
        signal.set_wakeup_fd(-1)
    except ValueError:
        # forked from a non-main thread, where the signal module refuses to
        # act; closing our copy of the pipe still protects the parent
        if _wakeup_write_fd is not None:
            os.close(_wakeup_write_fd)


def _ignore_signal(signum: int, frame: types.FrameType | None) -> None:
    # exists so CPython delivers the signal (writing it to the wakeup fd)
    # instead of taking the default action; the listener thread does the work
    pass


def _write_stderr(message: str) -> None:
    # os.write, not the logger: the main thread may hold the logging lock,
    # wedged in the collective the signal interrupted
    os.write(2, message.encode())


def _wait_for_termination_signal(read_fd: int) -> signal.Signals | None:
    while True:
        data = os.read(read_fd, 64)
        if not data or _STOP_LISTENING in data:
            return None
        # the wakeup fd sees every signal CPython delivers (torch's DataLoader
        # installs a SIGCHLD handler, for one); act only on termination
        for sig in TERMINATION_SIGNALS:
            if sig in data:
                return sig


def _listen(read_fd: int, abort: Callable[[], None], grace_period: float) -> None:
    signum = _wait_for_termination_signal(read_fd)
    if signum is None:  # the context exited normally
        return
    _write_stderr(
        f"Received {signal.Signals(signum).name}, aborting distributed "
        "communicators before exiting.\n"
    )
    try:
        abort()
    except BaseException:
        _write_stderr(f"Aborting communicators failed:\n{traceback.format_exc()}")
    # peers' aborts must finish, so their kernels stop touching our memory,
    # before we exit
    time.sleep(grace_period)
    _write_stderr(f"Exiting with code {128 + signum}.\n")
    os._exit(128 + signum)


@contextlib.contextmanager
def handle_termination_signals(
    abort: Callable[[], None],
    grace_period: float = DEFAULT_GRACE_PERIOD,
) -> Generator[None, None, None]:
    """Exit on SIGTERM or SIGINT, aborting the distributed backend first.

    Args:
        abort: Locally aborts the backend's communicators. Called on the
            listener thread, typically while the main thread is blocked in a
            collective, so it must not require the main thread's cooperation.
        grace_period: Seconds to wait between aborting and exiting, so that
            peers' aborts finish while this process's memory is still mapped.
    """
    global _armed, _wakeup_write_fd, _at_fork_registered
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

    if not _at_fork_registered:
        os.register_at_fork(after_in_child=_disarm_in_child)
        _at_fork_registered = True

    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    listener = threading.Thread(
        target=_listen,
        args=(read_fd, abort, grace_period),
        name="termination-listener",
        daemon=True,
    )
    listener.start()
    previous_handlers = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    previous_fd = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    for sig in TERMINATION_SIGNALS:
        signal.signal(sig, _ignore_signal)
    _wakeup_write_fd = write_fd
    _armed = True
    try:
        yield
    finally:
        _armed = False
        _wakeup_write_fd = None
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        signal.set_wakeup_fd(previous_fd)
        os.write(write_fd, bytes([_STOP_LISTENING]))
        listener.join(timeout=10.0)
        os.close(write_fd)
        os.close(read_fd)
