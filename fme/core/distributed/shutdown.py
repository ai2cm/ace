"""Tear the distributed backend down safely when a job is preempted.

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

A Python signal handler runs only when the main thread returns to the
interpreter, and a preempted rank's main thread is typically blocked inside a
collective's stream sync, so a handler could never run in time. Instead the
signal is observed on a dedicated listener thread through
``signal.set_wakeup_fd``, which CPython's C-level handler writes regardless of
what the main thread is doing.

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

# How long the listener waits for the main thread to unwind into the
# context's exit before giving up on the post-abort callbacks. One batch of
# compute bounds the unwind in the common case; together with the callbacks
# and grace period this must fit torchrun's 30s SIGTERM-to-SIGKILL budget.
DEFAULT_PARK_TIMEOUT = 5.0

# tells the sentinel apart from signal numbers, which are all positive
_STOP_LISTENING = 0

_armed = False  # whether the wakeup fd currently belongs to this module
_pipe_fds: tuple[int, int] | None = None
_at_fork_registered = False

_post_abort_callbacks: list[Callable[[], None]] = []


def add_post_abort_callback(callback: Callable[[], None]) -> None:
    """Run ``callback`` on the listener thread after the abort, before exit.

    Callbacks run only once the main thread has unwound into the context's
    exit, where it blocks until the process ends -- so their reads of training
    state cannot race it. A main thread that has not unwound within
    ``park_timeout`` (still computing, or stuck somewhere that is not a
    collective) forfeits the callbacks rather than risk a torn snapshot.
    Best-effort in duration too: the scheduler's SIGKILL caps how long a
    callback may take. The communicators are gone by then, so callbacks must
    not use collectives, nor the logging module (see ``write_stderr``).
    """
    _post_abort_callbacks.append(callback)


def clear_post_abort_callbacks() -> None:
    _post_abort_callbacks.clear()


def _disarm_in_child() -> None:
    """Undo the parent's signal setup in a forked child.

    A fork-started DataLoader worker inherits the ignore-handler and the wakeup
    fd, which still feeds the parent's pipe: a signal delivered to the worker
    would masquerade as the parent's own preemption. Restore the defaults so
    the child dies as it would have without the inheritance. This runs in the
    child's main thread -- fork's surviving thread is re-designated as such
    before the at-fork hooks run -- so the signal module cooperates.
    """
    global _armed, _pipe_fds
    if not _armed:
        return
    for sig in TERMINATION_SIGNALS:
        signal.signal(sig, signal.SIG_DFL)
    signal.set_wakeup_fd(-1)
    if _pipe_fds is not None:
        for fd in _pipe_fds:
            os.close(fd)
    # clear the state, or a second-generation fork (dataset code forking a
    # worker's own subprocess) re-runs this on fd numbers the worker has
    # since reused
    _armed = False
    _pipe_fds = None


def _ignore_signal(signum: int, frame: types.FrameType | None) -> None:
    # exists so CPython delivers the signal (writing it to the wakeup fd)
    # instead of taking the default action; the listener thread does the work
    pass


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


def _wait_for_termination_signal(read_fd: int) -> signal.Signals | None:
    while True:
        data = os.read(read_fd, 64)
        # the wakeup fd sees every signal CPython delivers (torch's DataLoader
        # installs a SIGCHLD handler, for one); act only on termination.
        # Checked before the sentinel: a signal racing the context's exit must
        # win, or the process walks on with its protection already removed.
        for sig in TERMINATION_SIGNALS:
            if sig in data:
                return sig
        if not data or _STOP_LISTENING in data:
            return None


def _listen(
    read_fd: int,
    abort: Callable[[], None],
    grace_period: float,
    park_timeout: float,
    terminating: threading.Event,
    main_parked: threading.Event,
) -> None:
    signum = _wait_for_termination_signal(read_fd)
    if signum is None:  # the context exited normally
        return
    terminating.set()  # from here on this thread owns the process's exit
    write_stderr(
        f"Received {signal.Signals(signum).name}, aborting distributed "
        "communicators before exiting.\n"
    )
    try:
        # not bounded: if the abort hangs, exiting anyway would guarantee the
        # peer-GPU fault, so the rank rides to the scheduler's SIGKILL instead
        abort()
    except BaseException:
        write_stderr(f"Aborting communicators failed:\n{traceback.format_exc()}")
    if _post_abort_callbacks:
        # the abort releases only a main thread blocked in a collective; one
        # between collectives keeps computing until the next collective
        # raises, and the callbacks' reads of training state would race it.
        # Wait for it to block in the context's exit, and forfeit the
        # callbacks rather than record a torn snapshot if it never does.
        if main_parked.wait(park_timeout):
            for callback in _post_abort_callbacks:
                try:
                    callback()
                except BaseException:
                    write_stderr(
                        f"Post-abort callback failed:\n{traceback.format_exc()}"
                    )
        else:
            write_stderr(
                f"Main thread still running {park_timeout}s after the abort; "
                "skipping post-abort callbacks.\n"
            )
    # peers' aborts must finish, so their kernels stop touching our memory,
    # before we exit
    time.sleep(grace_period)
    write_stderr(f"Exiting with code {128 + signum}.\n")
    os._exit(128 + signum)


@contextlib.contextmanager
def handle_termination_signals(
    abort: Callable[[], None],
    grace_period: float = DEFAULT_GRACE_PERIOD,
    park_timeout: float = DEFAULT_PARK_TIMEOUT,
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
        park_timeout: Seconds the listener waits for the main thread to block
            in this context's exit before skipping the post-abort callbacks
            (see ``add_post_abort_callback``).
    """
    global _armed, _pipe_fds, _at_fork_registered
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
    terminating = threading.Event()
    main_parked = threading.Event()
    listener = threading.Thread(
        target=_listen,
        args=(read_fd, abort, grace_period, park_timeout, terminating, main_parked),
        name="termination-listener",
        daemon=True,
    )
    listener.start()
    previous_handlers = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    previous_fd = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    for sig in TERMINATION_SIGNALS:
        signal.signal(sig, _ignore_signal)
    _pipe_fds = (read_fd, write_fd)
    _armed = True
    try:
        yield
    finally:
        _armed = False
        _pipe_fds = None
        # the main thread only joins the listener and restores dispositions
        # from here; it will not touch training state again, so the
        # post-abort callbacks may read it
        main_parked.set()
        # a signal delivered between the listener consuming this sentinel and
        # the dispositions being restored below is dropped; the window is
        # microseconds wide and the process is exiting anyway
        os.write(write_fd, bytes([_STOP_LISTENING]))
        if not terminating.is_set():
            # covers the race where a signal was delivered but not yet read:
            # long enough for the listener to abort and exit
            listener.join(timeout=grace_period + 10.0)
        if terminating.is_set():
            # the listener owns the exit: hold the unwinding main thread for
            # the abort and grace period, with the ignore-handlers still
            # installed -- restoring the previous dispositions (typically
            # SIG_DFL) here would let a repeated signal kill the rank before
            # the grace period ends. If the abort hangs, the scheduler's
            # SIGKILL is the backstop.
            listener.join()
        else:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            signal.set_wakeup_fd(previous_fd)
            os.close(write_fd)
            os.close(read_fd)
            # the callbacks belonged to this context's session; a later
            # context must not fire them against torn-down state
            clear_post_abort_callbacks()
