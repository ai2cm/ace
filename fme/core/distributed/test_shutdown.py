import os
import select
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch.utils.data

from fme.core.distributed.shutdown import (
    add_post_shutdown_callback,
    handle_termination_signals,
)


def _run_handler_program(program: str) -> "subprocess.CompletedProcess[str]":
    """Exercise the handler in a subprocess.

    Cases that turn on whether the process survives cannot run in-process:
    pytest catches the graceful `SystemExit`, and an `os._exit` would take the
    whole session down without a summary.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        timeout=60,
        text=True,
    )


def test_process_group_is_torn_down_before_callbacks_run():
    """Slow work must not delay the collective teardown.

    `destroy_process_group` is collective, so a rank that writes a checkpoint
    first leaves its peers blocked in the collective until the scheduler kills
    them mid-NCCL, which is what faults the GPUs.
    """
    events = []
    add_post_shutdown_callback(lambda: events.append("first callback"))
    add_post_shutdown_callback(lambda: events.append("second callback"))

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)

    assert events == ["shutdown", "first callback", "second callback"]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_exits_with_conventional_code_for_signal(sig):
    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit) as excinfo:
            signal.raise_signal(sig)

    assert excinfo.value.code == 128 + sig


def test_callbacks_run_even_if_shutdown_fails():
    """A backend that cannot be torn down must not cost us the checkpoint."""
    events = []
    add_post_shutdown_callback(lambda: events.append("callback"))

    def shutdown():
        raise RuntimeError("no process group")

    with handle_termination_signals(shutdown=shutdown):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)

    assert events == ["callback"]


def test_later_callbacks_run_even_if_an_earlier_one_fails():
    events = []

    def failing():
        raise RuntimeError("boom")

    add_post_shutdown_callback(failing)
    add_post_shutdown_callback(lambda: events.append("callback"))

    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)

    assert events == ["callback"]


def test_previous_handlers_and_callbacks_are_restored_on_exit():
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    ran = []

    with handle_termination_signals(shutdown=lambda: None):
        add_post_shutdown_callback(lambda: ran.append("callback"))
        assert signal.getsignal(signal.SIGTERM) is not original[signal.SIGTERM]

    for sig, handler in original.items():
        assert signal.getsignal(sig) is handler

    # the callback belonged to the job that just ended, so a later job must not
    # inherit it
    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)
    assert ran == []


def test_a_second_signal_does_not_restart_the_teardown():
    """A repeated Ctrl-C must not cost us the restart checkpoint.

    Re-entering would abandon the in-progress teardown and start it over, so
    the callbacks would run twice and the checkpoint write would restart.
    """
    events = []

    def shutdown():
        events.append("shutdown")
        signal.raise_signal(signal.SIGINT)  # arrives while we are tearing down

    add_post_shutdown_callback(lambda: events.append("callback"))

    with handle_termination_signals(shutdown=shutdown):
        with pytest.raises(SystemExit) as excinfo:
            signal.raise_signal(signal.SIGTERM)

    assert events == ["shutdown", "callback"]
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_a_signal_after_a_swallowed_exit_kills_the_process(monkeypatch):
    """A process that swallowed the teardown's SystemExit must stay killable.

    pytest turns a SystemExit raised inside a test into a test failure and
    keeps running (as does any bare except), and the ignore-repeats guard
    would otherwise latch every later signal into a no-op: the first Ctrl-C
    of a test session would make the rest of it un-interruptible.
    """
    exited = []
    monkeypatch.setattr(os, "_exit", lambda code: exited.append(code))

    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)
        # the SystemExit was swallowed just above; the job is still running
        signal.raise_signal(signal.SIGINT)

    assert exited == [128 + signal.SIGINT]


def test_callback_raising_system_exit_does_not_hijack_the_exit_code():
    """sys.exit() in a callback must not skip the rest or rewrite the code."""
    events = []
    add_post_shutdown_callback(lambda: sys.exit(0))
    add_post_shutdown_callback(lambda: events.append("callback"))

    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit) as excinfo:
            signal.raise_signal(signal.SIGTERM)

    assert events == ["callback"]
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_deadline_does_not_outlive_a_successful_teardown(monkeypatch):
    """An expired deadline must not kill a process that shut down cleanly.

    The timer runs on its own thread, so leaving it armed takes the process
    down some seconds after the graceful path already finished.

    This covers the common case, not an absolute guarantee: `Timer.cancel` is a
    no-op once the timer has begun running, so a clean teardown that finishes in
    the same instant the deadline expires still exits 143. Cancelling as soon as
    the collective returns keeps that window down to the collective's own
    duration -- it used to span the checkpoint write as well -- but it cannot
    close it.
    """
    exited = []
    monkeypatch.setattr(os, "_exit", lambda code: exited.append(code))

    with handle_termination_signals(shutdown=lambda: None, teardown_timeout=0.05):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)

    time.sleep(0.2)  # long enough that an uncancelled deadline would have fired
    assert exited == []


@pytest.mark.medium_duration
def test_hard_exits_when_teardown_hangs():
    """A wedged rank must not hold the process until the scheduler SIGKILLs it.

    Run in a subprocess: the point of the timeout is that it fires from another
    thread while the main thread is stuck, and the outcome is process death.
    """
    result = _run_handler_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import handle_termination_signals

        def hang():
            threading.Event().wait()

        with handle_termination_signals(shutdown=hang, teardown_timeout=1.0):
            signal.raise_signal(signal.SIGTERM)
        """
    )

    assert result.returncode == 128 + signal.SIGTERM
    assert "did not complete" in result.stderr


@pytest.mark.medium_duration
@pytest.mark.parametrize(
    "shutdown_body",
    [
        pytest.param("pass", id="collective_succeeded"),
        pytest.param(
            'raise RuntimeError("NCCL communicator was aborted")',
            id="collective_failed",
        ),
    ],
)
def test_the_deadline_does_not_bound_the_checkpoint_write(tmp_path, shutdown_body):
    """Rank 0 must finish its work whatever became of its peers.

    The deadline is there to stop a wedged rank holding its peers inside the
    collective. Once `shutdown` has returned they are out of it either way --
    cleanly, or because the communicator was already aborted -- so nothing is
    left for the deadline to protect, and leaving it armed can only truncate the
    restart checkpoint that running the teardown first exists to make room for.

    Both parameters therefore expect the same outcome: the write completes even
    though it outlasts `teardown_timeout` several times over.

    A subprocess, because the failure is `os._exit` mid-write. The exit code
    cannot distinguish it -- the graceful `sys.exit(143)` and the watchdog's
    `os._exit(143)` are identical -- so the evidence is the file plus stderr.
    """
    checkpoint = tmp_path / "checkpoint"
    result = _run_handler_program(
        f"""
        import signal, time
        from fme.core.distributed.shutdown import (
            add_post_shutdown_callback,
            handle_termination_signals,
        )

        def shutdown():
            {shutdown_body}

        def write_checkpoint():
            # stands in for the Trainer's multi-GB torch.save
            with open({str(checkpoint)!r}, "w") as f:
                f.write("partial")
            time.sleep(2.0)
            with open({str(checkpoint)!r}, "a") as f:
                f.write(" complete")

        add_post_shutdown_callback(write_checkpoint)
        with handle_termination_signals(shutdown=shutdown, teardown_timeout=0.5):
            signal.raise_signal(signal.SIGTERM)
        """
    )

    assert "did not complete" not in result.stderr
    assert checkpoint.read_text() == "partial complete"


@pytest.mark.medium_duration
def test_a_signal_during_the_callbacks_kills_the_process():
    """Once the peers are safe, an escalating signal must be obeyed.

    The callbacks are deliberately unbounded, so ignoring repeats here -- as we
    do during the collective -- would leave a rank wedged in a stalled
    checkpoint write deaf to every signal until SIGKILL. By this point the
    process group is gone, so honoring the signal costs at most the checkpoint.
    """
    result = _run_handler_program(
        """
        import os, signal
        from fme.core.distributed.shutdown import (
            add_post_shutdown_callback,
            handle_termination_signals,
        )

        def stalled_write():
            print("first callback", flush=True)  # os._exit will not flush for us
            signal.raise_signal(signal.SIGINT)  # the operator, losing patience
            print("first callback returned", flush=True)

        add_post_shutdown_callback(stalled_write)
        add_post_shutdown_callback(lambda: print("second callback", flush=True))
        with handle_termination_signals(shutdown=lambda: None):
            signal.raise_signal(signal.SIGTERM)
        """
    )

    # died inside the first callback: the second never ran, and the exit code is
    # the escalating signal's rather than the original SIGTERM's
    assert result.stdout.splitlines() == ["first callback"]
    assert result.returncode == 128 + signal.SIGINT


def test_no_handler_installed_off_the_main_thread():
    """Only the main thread can own the process's signal disposition."""
    original = signal.getsignal(signal.SIGTERM)
    observed = []

    def run():
        with handle_termination_signals(shutdown=lambda: None):
            observed.append(signal.getsignal(signal.SIGTERM))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert observed == [original]


def test_no_teardown_in_a_process_forked_inside_the_context():
    """A forked child must not tear the parent's backend down.

    The default DataLoader start method is fork, so each worker inherits this
    handler, and torchrun signals the whole process group. Every worker would
    then destroy a fork-inherited NCCL context and re-run the parent's
    callbacks -- including the multi-GB restart checkpoint write.

    The `get_worker_info` guard cannot catch this: it is only set in workers,
    which under fork have already inherited the handler by the time it runs,
    and spawn/forkserver workers never inherited it at all.
    """
    read_fd, write_fd = os.pipe()

    def report(event: bytes) -> None:
        # the child cannot report back through the parent's objects
        os.write(write_fd, event)

    add_post_shutdown_callback(lambda: report(b"callback"))

    with handle_termination_signals(shutdown=lambda: report(b"shutdown")):
        pid = os.fork()
        if pid == 0:
            # never let a forked copy of the test session escape back into
            # pytest, whatever the handler does
            try:
                os.close(read_fd)
                signal.raise_signal(signal.SIGTERM)
            except BaseException:
                pass
            finally:
                os._exit(0)

        os.close(write_fd)
        ready: list[int] = []
        try:
            # EOF (the child exiting) or its first write, whichever comes first
            ready, _, _ = select.select([read_fd], [], [], 30.0)
            observed = os.read(read_fd, 4096) if ready else b"child hung"
        finally:
            os.close(read_fd)
            if not ready:
                os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)

    assert observed == b""
    # and the other half of the guard's contract: the child dies from the signal
    # as it would have without inheriting the handler. Without this the child's
    # `os._exit(0)` backstop would let a `raise_signal` that quietly failed to
    # kill anything pass as success.
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGTERM


def test_no_handler_installed_in_a_dataloader_worker(monkeypatch):
    """The DataLoader owns its workers' lifecycle, so leave their signals alone."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    original = signal.getsignal(signal.SIGTERM)

    with handle_termination_signals(shutdown=lambda: None):
        assert signal.getsignal(signal.SIGTERM) is original
