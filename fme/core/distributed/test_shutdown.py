import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch.utils.data

from fme.core.distributed.shutdown import (
    TERMINATION_SIGNALS,
    handle_termination_signals,
)


def _run_listener_program(program: str) -> "subprocess.CompletedProcess[str]":
    """Exercise the listener in a subprocess.

    Every acted-on signal ends in `os._exit`, which would take the whole pytest
    session down if raised in-process.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        timeout=60,
        text=True,
    )


@pytest.mark.medium_duration
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_aborts_then_exits_with_conventional_code_for_signal(sig):
    result = _run_listener_program(
        f"""
        import signal, threading
        from fme.core.distributed.shutdown import handle_termination_signals

        with handle_termination_signals(
            abort=lambda: print("abort", flush=True), grace_period=0.1
        ):
            signal.raise_signal(signal.Signals({int(sig)}))
            threading.Event().wait()  # the listener must end the process
        """
    )

    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + sig


@pytest.mark.medium_duration
def test_acts_while_the_main_thread_is_wedged_holding_the_logging_lock():
    """The case the old handler-based design could not cover.

    A rank blocked in a collective's stream sync never returns to the
    interpreter, so a Python signal handler never runs; and it passes through
    `Handler.handle`'s locked emit constantly, so a signal arriving there
    leaves the main thread holding the logging lock as it wedges. The listener
    must act anyway, without needing that lock.
    """
    result = _run_listener_program(
        """
        import logging, os, signal, sys, threading, time
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        from fme.core.distributed.shutdown import handle_termination_signals

        def signal_later():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=signal_later, daemon=True).start()
        lock = threading.Lock()
        with handle_termination_signals(
            abort=lambda: print("abort", flush=True), grace_period=0.1
        ):
            # stands in for being interrupted mid-emit, inside `Handler.handle`
            with logging.getLogger().handlers[0].lock:
                lock.acquire()
                lock.acquire()  # blocks in C, never returning to the interpreter
        """
    )

    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_exits_even_if_abort_raises():
    """A backend that cannot be aborted must not leave the rank running.

    Riding to the scheduler's SIGKILL is the outcome this module exists to
    avoid, so a failed abort still exits after the grace period.
    """
    result = _run_listener_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import handle_termination_signals

        def abort():
            raise RuntimeError("no process group")

        with handle_termination_signals(abort=abort, grace_period=0.1):
            signal.raise_signal(signal.SIGTERM)
            threading.Event().wait()
        """
    )

    assert result.returncode == 128 + signal.SIGTERM
    assert "RuntimeError: no process group" in result.stderr


@pytest.mark.medium_duration
def test_exit_waits_out_the_grace_period():
    """No rank may exit before its peers' aborts have finished.

    The grace period after the local abort is the only thing standing between
    a fast rank's exit and a slow peer's still-running kernels, so the exit
    must not come early.
    """
    grace_period = 1.0
    program = textwrap.dedent(
        f"""
        import signal, threading
        from fme.core.distributed.shutdown import handle_termination_signals

        with handle_termination_signals(
            abort=lambda: None, grace_period={grace_period}
        ):
            print("ready", flush=True)
            threading.Event().wait()
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        start = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        elapsed = time.monotonic() - start
    finally:
        proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    assert proc.returncode == 128 + signal.SIGTERM
    assert elapsed >= grace_period * 0.9


@pytest.mark.medium_duration
def test_a_main_thread_released_by_the_abort_cannot_exit_first():
    """Leaving the context must block until the listener has exited.

    Aborting the communicators *releases* a main thread that was blocked in a
    collective, which then raises out of the training loop and unwinds. If
    that unwind escaped the context, the process would exit with the
    exception's code, before the grace period -- losing both the fabric
    guarantee and the preemption exit code.
    """
    result = _run_listener_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import handle_termination_signals

        released = threading.Event()

        def abort():
            print("abort", flush=True)
            released.set()

        with handle_termination_signals(abort=abort, grace_period=1.0):
            signal.raise_signal(signal.SIGTERM)
            released.wait()  # blocked "in the collective" until the abort
            raise RuntimeError("rank unwinding after its collective died")
        """
    )

    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_forked_child_does_not_trigger_the_parent_abort():
    """A signal delivered to a fork-started worker must stay the worker's.

    The default DataLoader start method is fork, so each worker inherits the
    wakeup fd, which feeds the parent's pipe: without the at-fork disarm, a
    SIGTERM that the DataLoader sends a worker would read as the parent's own
    preemption and abort the whole rank. The disarmed child dies from the
    signal as it would have without the inheritance.
    """
    result = _run_listener_program(
        """
        import os, signal, time
        from fme.core.distributed.shutdown import handle_termination_signals

        with handle_termination_signals(
            abort=lambda: print("abort", flush=True), grace_period=0.1
        ):
            pid = os.fork()
            if pid == 0:
                time.sleep(30)
                os._exit(0)
            time.sleep(0.5)  # let the child's at-fork disarm run
            os.kill(pid, signal.SIGTERM)
            _, status = os.waitpid(pid, 0)
            child_killed = (
                os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGTERM
            )
            print(f"child killed by signal: {child_killed}", flush=True)
            time.sleep(0.5)  # a poisoned listener would abort and exit here
            print("parent alive", flush=True)
        """
    )

    assert result.stdout.splitlines() == [
        "child killed by signal: True",
        "parent alive",
    ]
    assert result.returncode == 0


@pytest.mark.medium_duration
def test_other_handled_signals_do_not_trigger_the_abort():
    """The wakeup fd sees every signal CPython delivers, not just termination.

    torch's DataLoader installs a SIGCHLD handler in the main process, so
    routine worker exits write to the same pipe the listener reads; acting on
    those bytes would abort a healthy rank whenever a worker pool shut down.
    """
    result = _run_listener_program(
        """
        import os, signal, time
        from fme.core.distributed.shutdown import handle_termination_signals

        signal.signal(signal.SIGCHLD, lambda signum, frame: None)
        with handle_termination_signals(
            abort=lambda: print("abort", flush=True), grace_period=0.1
        ):
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            time.sleep(0.5)
            print("alive", flush=True)
        """
    )

    assert result.stdout.splitlines() == ["alive"]
    assert result.returncode == 0


def test_previous_dispositions_are_restored_on_exit():
    original = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}

    with handle_termination_signals(abort=lambda: None):
        for sig in TERMINATION_SIGNALS:
            assert signal.getsignal(sig) is not original[sig]
        assert any(
            thread.name == "termination-listener" for thread in threading.enumerate()
        )

    for sig, handler in original.items():
        assert signal.getsignal(sig) is handler
    # the sentinel write on exit stops the listener rather than leaking one
    # thread per context entry
    assert not any(
        thread.name == "termination-listener" for thread in threading.enumerate()
    )


def test_no_listener_installed_off_the_main_thread():
    """Only the main thread can own the process's signal disposition."""
    original = signal.getsignal(signal.SIGTERM)
    observed = []

    def run():
        with handle_termination_signals(abort=lambda: None):
            observed.append(signal.getsignal(signal.SIGTERM))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert observed == [original]


def test_no_listener_installed_in_a_dataloader_worker(monkeypatch):
    """The DataLoader owns its workers' lifecycle, so leave their signals alone."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    original = signal.getsignal(signal.SIGTERM)

    with handle_termination_signals(abort=lambda: None):
        assert signal.getsignal(signal.SIGTERM) is original
