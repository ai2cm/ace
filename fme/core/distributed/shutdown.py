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
rides to the scheduler's SIGKILL. The watchdog thread bounds a teardown once
it has begun; it cannot start one. Likewise the instants before the handler is
installed -- ``init_process_group`` itself, inside ``Distributed.context()``
entry -- remain unprotected.
"""

import contextlib
import logging
import os
import signal
import sys
import threading
import types
from collections.abc import Callable, Generator

import torch.utils.data

logger = logging.getLogger(__name__)

TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)

# Bounds the collective teardown. Beaker's grace period and torchrun's elastic
# agent (30s) both start counting when the signal is delivered, so a rank that
# is wedged in an unrelated collective must not be able to hold the others past
# the shorter of the two.
DEFAULT_TEARDOWN_TIMEOUT = 20.0

_post_shutdown_callbacks: list[Callable[[], None]] = []


def add_post_shutdown_callback(callback: Callable[[], None]) -> None:
    """Register work to run on termination, once the process group is gone.

    Callbacks run in registration order and must not use collectives. They run
    on borrowed time -- the scheduler may SIGKILL the process at any point
    after its grace period -- so they are best-effort, and the most valuable
    work should be registered first.
    """
    _post_shutdown_callbacks.append(callback)


def clear_post_shutdown_callbacks() -> None:
    """Discard every registered callback.

    The registry is process-global, so tests that build objects which register
    callbacks need to reset it between cases.
    """
    _post_shutdown_callbacks.clear()


def _hard_exit_after(timeout: float, exit_code: int) -> threading.Timer:
    """Exit the process if the graceful teardown has not finished in time.

    A blocked collective does not return to the interpreter, so a Python signal
    handler or ``signal.alarm`` would never run; only a separate thread can
    enforce the deadline.
    """

    def give_up() -> None:
        logger.error(
            "Distributed shutdown did not complete within %.0fs, exiting now. "
            "GPUs on this node may need to be reset.",
            timeout,
        )
        os._exit(exit_code)

    timer = threading.Timer(timeout, give_up)
    timer.daemon = True
    timer.start()
    return timer


def _tear_down(shutdown: Callable[[], None]) -> None:
    """Release the backend, then run the best-effort callbacks.

    Nothing here may raise: every remaining step is worth attempting even if an
    earlier one failed.
    """
    try:
        shutdown()
    except BaseException:
        logger.exception("Failed to shut down the distributed backend.")
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
) -> Generator[None, None, None]:
    """Shut the distributed backend down before exiting on SIGTERM or SIGINT.

    Args:
        shutdown: Tears the distributed backend down. Called before any
            callback registered with `add_post_shutdown_callback`, so that
            every rank reaches the collective teardown together.
        teardown_timeout: Seconds to allow for `shutdown` and the callbacks
            before exiting regardless.
    """
    if threading.current_thread() is not threading.main_thread():
        # only the main thread may install handlers; a thread shares the
        # process disposition its main thread installed
        yield
        return
    if torch.utils.data.get_worker_info() is not None:
        # DataLoader workers enter this context only to learn their rank. The
        # DataLoader owns their lifecycle, so leave their signal disposition to
        # it rather than exiting out from under it.
        yield
        return

    installed_pid = os.getpid()
    tearing_down = False
    teardown_complete = False

    def handle(signum: int, frame: types.FrameType | None) -> None:
        nonlocal tearing_down, teardown_complete
        exit_code = 128 + signum
        if os.getpid() != installed_pid:
            # A forked child inherited this handler: the default DataLoader
            # start method is fork, and the scheduler signals the whole process
            # group, so every worker would otherwise destroy a fork-inherited
            # process group and re-run the parent's callbacks -- including the
            # restart-checkpoint write, racing the parent's write of the same
            # path. Only the process that installed the handler owns the
            # teardown; die as this process would have without the inheritance.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        if teardown_complete:
            # the SystemExit from the first signal was swallowed (pytest turns
            # it into a test failure and keeps running; so does any bare
            # except), so being here means graceful exit failed. Honor the
            # convention that a repeated signal kills the process.
            logger.info(
                "Received %s after teardown already completed; exiting.",
                signal.Signals(signum).name,
            )
            os._exit(exit_code)
        if tearing_down:
            # a repeated Ctrl-C, or both the scheduler and torchrun signalling.
            # The first handler owns the teardown and the deadline already
            # bounds it, so restarting it here would only cost us the callbacks.
            logger.info(
                "Received %s while already shutting down; ignoring.",
                signal.Signals(signum).name,
            )
            return
        tearing_down = True
        logger.info(
            "Received %s, shutting down the distributed backend before exiting.",
            signal.Signals(signum).name,
        )
        deadline = _hard_exit_after(teardown_timeout, exit_code)
        try:
            _tear_down(shutdown)
        finally:
            teardown_complete = True
            # the deadline bounds the teardown, so it must not outlive it and
            # kill a process that already shut down cleanly
            deadline.cancel()
        sys.exit(exit_code)

    previous = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    for sig in TERMINATION_SIGNALS:
        signal.signal(sig, handle)
    try:
        yield
    finally:
        for sig, previous_handler in previous.items():
            signal.signal(sig, previous_handler)
        clear_post_shutdown_callbacks()
