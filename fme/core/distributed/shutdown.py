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
"""

import contextlib
import enum
import logging
import os
import signal
import sys
import threading
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

_post_shutdown_callbacks: list[Callable[[], None]] = []


class _Phase(enum.Enum):
    """How far the teardown has got, which is what a repeated signal turns on."""

    RUNNING = enum.auto()
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


def _hard_exit_after(timeout: float, exit_code: int) -> threading.Timer:
    """Exit the process if the graceful teardown has not finished in time.

    A blocked collective does not return to the interpreter, so a Python signal
    handler or ``signal.alarm`` would never run; only a separate thread can
    enforce the deadline.
    """

    def give_up() -> None:
        # `os.write` rather than the logger: a handler holds its lock across an
        # emit, so if the signal arrived inside that window the main thread holds
        # it and is now wedged below. Acquiring it from this thread would block
        # for good, leaving the rank to be SIGKILLed.
        os.write(
            2,
            f"Distributed shutdown did not complete within {timeout:.0f}s, "
            "exiting now. GPUs on this node may need to be reset.\n".encode(),
        )
        os._exit(exit_code)

    timer = threading.Timer(timeout, give_up)
    timer.daemon = True
    timer.start()
    return timer


def _shut_down_backend(shutdown: Callable[[], None], deadline: threading.Timer) -> None:
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
) -> Generator[None, None, None]:
    """Shut the distributed backend down before exiting on SIGTERM or SIGINT.

    Args:
        shutdown: Tears the distributed backend down. Called before any
            callback registered with `add_post_shutdown_callback`, so that
            every rank reaches the collective teardown together.
        teardown_timeout: Seconds to allow `shutdown` before exiting
            regardless. It does not bound the callbacks that follow; those are
            left to run against the scheduler's clock.
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

    def handle(signum: int, frame: types.FrameType | None) -> None:
        nonlocal phase
        exit_code = 128 + signum
        if os.getpid() != installed_pid:
            # A forked child inherited this handler: the default DataLoader start
            # method is fork and the scheduler signals the whole process group, so
            # every worker would otherwise tear down a fork-inherited process
            # group and re-run the parent's callbacks, racing its checkpoint
            # write. Die as this process would have without the inheritance.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
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
        phase = _Phase.COLLECTIVE
        logger.info(
            "Received %s, shutting down the distributed backend before exiting.",
            signal.Signals(signum).name,
        )
        _shut_down_backend(shutdown, _hard_exit_after(teardown_timeout, exit_code))
        phase = _Phase.CALLBACKS
        try:
            _run_post_shutdown_callbacks()
        finally:
            phase = _Phase.COMPLETE
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
