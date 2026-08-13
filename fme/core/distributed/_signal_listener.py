"""Deliver signals to a dedicated thread, whatever the main thread is doing.

A Python signal handler runs only when the main thread returns to the
interpreter, and a preempted rank's main thread is typically blocked inside a
collective's stream sync, so a handler could never run in time. Instead,
CPython's C-level handler writes each delivered signal number to a wakeup fd
(``signal.set_wakeup_fd``; an fd, or file descriptor, is the integer handle
the OS gives an open file or pipe). `SignalListener` points that fd at a pipe
and reads it from its own thread, so the signal is observed regardless of
what the main thread is doing.

This module is pure delivery mechanism: what to *do* about a termination
signal is `fme.core.distributed.shutdown`'s concern.
"""

import os
import signal
import threading
import types
from collections.abc import Callable
from typing import Any

# what signal.getsignal returns and signal.signal accepts
_SignalHandler = (
    Callable[[int, types.FrameType | None], Any] | int | signal.Handlers | None
)

# tells the sentinel apart from signal numbers, which are all positive
_STOP_LISTENING = 0

# The wakeup fd and signal dispositions are process-global, so at most one
# listener is active, and the at-fork hook must be able to find it.
_active_listener: "SignalListener | None" = None
_at_fork_registered = False


def _route_to_listener(signum: int, frame: types.FrameType | None) -> None:
    # exists so CPython delivers the signal (writing it to the wakeup fd)
    # instead of taking the default action; the listener thread does the work
    pass


def _disarm_in_child() -> None:
    """Undo the parent's signal setup in a forked child.

    A fork-started DataLoader worker inherits the routing handler and the
    wakeup fd, which still feeds the parent's pipe: a signal delivered to the
    worker would masquerade as the parent's own. Restore the defaults so the
    child dies as it would have without the inheritance. This runs in the
    child's main thread -- fork's surviving thread is re-designated as such
    before the at-fork hooks run -- so the signal module cooperates.
    """
    global _active_listener
    listener = _active_listener
    if listener is None:
        return
    for sig in listener._signals:
        signal.signal(sig, signal.SIG_DFL)
    signal.set_wakeup_fd(-1)
    os.close(listener._read_fd)
    os.close(listener._write_fd)
    # clear the state, or a second-generation fork (dataset code forking a
    # worker's own subprocess) re-runs this on fd numbers the worker has
    # since reused
    _active_listener = None


class SignalListener:
    """Run a callback on a dedicated thread when one of ``signals`` arrives.

    ``on_signal`` runs on the listener's own thread, so it must not assume the
    main thread's cooperation -- and if it never returns (it may end the
    process), the thread never finishes. While the listener is armed the
    process's own disposition for ``signals`` is a no-op handler: nothing
    happens on delivery except the callback.

    The lifecycle is ``start()``, then exactly one of two teardowns, the
    caller choosing by whether a signal arrived:

    - no signal: ``request_stop()``, ``wait_until_finished()``, ``dismantle()``
      to restore the previous signal setup;
    - signal arrived: the listener stays armed -- so further signals stay
      no-ops -- and ``block_until_process_exit()`` parks the calling thread
      until ``on_signal`` ends the process.
    """

    def __init__(
        self,
        signals: tuple[signal.Signals, ...],
        on_signal: Callable[[signal.Signals], None],
    ) -> None:
        self._signals = signals
        self._on_signal = on_signal
        self._read_fd = -1
        self._write_fd = -1
        self._previous_handlers: dict[signal.Signals, _SignalHandler] = {}
        self._previous_wakeup_fd = -1
        self._thread = threading.Thread(
            target=self._read_pipe_until_signal_or_stop,
            name="termination-listener",
            daemon=True,
        )

    def start(self) -> None:
        """Arm the listener: point the wakeup fd at a fresh pipe, route
        ``signals`` to it, and start the thread that reads it. Main thread
        only (only it may install signal handlers).
        """
        global _active_listener, _at_fork_registered
        if not _at_fork_registered:
            os.register_at_fork(after_in_child=_disarm_in_child)
            _at_fork_registered = True
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._write_fd, False)
        self._thread.start()
        self._previous_handlers = {sig: signal.getsignal(sig) for sig in self._signals}
        self._previous_wakeup_fd = signal.set_wakeup_fd(
            self._write_fd, warn_on_full_buffer=False
        )
        for sig in self._signals:
            signal.signal(sig, _route_to_listener)
        _active_listener = self

    def _read_pipe_until_signal_or_stop(self) -> None:
        while True:
            data = os.read(self._read_fd, 64)
            # the wakeup fd sees every signal CPython delivers (torch's
            # DataLoader installs a SIGCHLD handler, for one); act only on
            # ours. Checked before the sentinel: a signal racing the stop
            # request must win, or the process walks on with its protection
            # already removed.
            for sig in self._signals:
                if sig in data:
                    self._on_signal(sig)
                    return
            if not data or _STOP_LISTENING in data:
                return

    def request_stop(self) -> None:
        """Ask the listener thread to finish, assuming no signal arrived.

        A signal already delivered but not yet read still wins over the stop
        request. One delivered between the thread consuming the sentinel and
        ``dismantle()`` restoring the dispositions is dropped; the window is
        microseconds wide and the process is exiting anyway.
        """
        global _active_listener
        _active_listener = None
        os.write(self._write_fd, bytes([_STOP_LISTENING]))

    def wait_until_finished(self, timeout: float) -> None:
        """Block up to ``timeout`` seconds for the listener thread to finish
        (``Thread.join``: wait for a thread to end).
        """
        self._thread.join(timeout=timeout)

    def block_until_process_exit(self) -> None:
        """Park the calling thread until ``on_signal`` ends the process.

        Never returns: the listener thread is running an ``on_signal`` that
        exits, and joining a thread that never finishes blocks forever. The
        process's external kill (the scheduler's SIGKILL) is the backstop if
        ``on_signal`` hangs.
        """
        self._thread.join()

    def dismantle(self) -> None:
        """Restore the pre-``start()`` signal setup and close the pipe.

        Only after ``wait_until_finished`` confirms no signal arrived: this
        removes the process's protection, and closes fds the listener thread
        would otherwise still be reading.
        """
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        signal.set_wakeup_fd(self._previous_wakeup_fd)
        os.close(self._write_fd)
        os.close(self._read_fd)
