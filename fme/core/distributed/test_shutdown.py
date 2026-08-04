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
    defer_termination,
    handle_termination_signals,
    write_marker,
)


def _marker_lines(event: str, captured: str) -> list[str]:
    prefix = f"fme-stop:{event} "
    return [line for line in captured.splitlines() if line.startswith(prefix)]


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
    """A wedged rank must not hold the process until the scheduler SIGKILLs it."""
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
def test_hard_exits_when_the_teardown_hangs_holding_the_logging_lock():
    """The watchdog must not need a lock the wedged main thread is holding.

    A rank passes through `Handler.handle`'s locked emit constantly, so a signal
    arriving there leaves the main thread holding that lock as it wedges in the
    collective. Anything the watchdog logs would then block for good.
    """
    result = _run_handler_program(
        """
        import logging, signal, sys, threading
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        from fme.core.distributed.shutdown import handle_termination_signals

        def hang():
            threading.Event().wait()

        # holding the lock stands in for being interrupted mid-emit, inside
        # `Handler.handle`'s `with self.lock`
        with handle_termination_signals(shutdown=hang, teardown_timeout=1.0):
            with logging.getLogger().handlers[0].lock:
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

    Once `shutdown` has returned the peers are out of the collective either way
    -- cleanly, or because the communicator was already aborted -- so both
    parameters expect the same outcome: the write completes even though it
    outlasts `teardown_timeout` several times over.

    The exit code cannot be the evidence, the graceful `sys.exit(143)` and the
    watchdog's `os._exit(143)` being identical, so it is the file plus stderr.
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


def test_signal_inside_a_deferral_does_not_shut_down(capfd):
    """A signal must not tear the backend down while a rendezvous is open.

    Tearing down from the handler is what strands peers: it happens wherever the
    signal landed, which on one rank is a batch boundary and on another the
    middle of a collective. Inside a deferral the signal is recorded instead, and
    the caller decides where to act on it.
    """
    events = []

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(SystemExit):
            with defer_termination(budget=1.0) as pending:
                signal.raise_signal(signal.SIGTERM)
                assert pending.requested
                assert pending.exit_code == 128 + signal.SIGTERM
                assert events == []

    assert events == ["shutdown"]
    deferred = _marker_lines("signal-deferred", capfd.readouterr().err)
    assert len(deferred) == 1
    assert "signal=SIGTERM" in deferred[0]


def test_deferred_stop_shuts_down_when_the_deferral_exits(capfd):
    """Leaving the deferral must run the teardown the handler would have run."""
    events = []
    add_post_shutdown_callback(lambda: events.append("first callback"))
    add_post_shutdown_callback(lambda: events.append("second callback"))

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(SystemExit) as excinfo:
            with defer_termination(budget=1.0):
                signal.raise_signal(signal.SIGTERM)

    assert events == ["shutdown", "first callback", "second callback"]
    assert excinfo.value.code == 128 + signal.SIGTERM
    # claim (b) of the evidence channel: this rank's teardown returned
    assert len(_marker_lines("shutdown-returned", capfd.readouterr().err)) == 1


def test_signal_outside_a_deferral_still_shuts_down_immediately():
    """Outside a loop there is no rendezvous to wait for, so nothing changes.

    Both sides of the deferral are checked: a signal before one has ever been
    entered, and a signal after one has been left, must each tear down from the
    handler as they do without this mechanism.
    """
    before = []
    with handle_termination_signals(shutdown=lambda: before.append("shutdown")):
        with pytest.raises(SystemExit) as excinfo:
            signal.raise_signal(signal.SIGTERM)
        assert before == ["shutdown"]
    assert excinfo.value.code == 128 + signal.SIGTERM

    after = []
    with handle_termination_signals(shutdown=lambda: after.append("shutdown")):
        with defer_termination(budget=1.0):
            pass  # left with nothing pending, so nothing was torn down
        assert after == []
        with pytest.raises(SystemExit) as excinfo:
            signal.raise_signal(signal.SIGTERM)
        assert after == ["shutdown"]
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_second_signal_while_a_stop_is_pending_is_ignored(monkeypatch, capfd):
    """A repeated SIGTERM must not defeat the deferral.

    In production a second SIGTERM is the norm rather than an escalation: the
    scheduler signals the container's process group and torchrun's agent then
    signals every rank again. Honouring it by tearing down immediately would
    strand peers on essentially every real preemption.
    """
    exited = []
    monkeypatch.setattr(os, "_exit", lambda code: exited.append(code))
    events = []

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(SystemExit):
            with defer_termination(budget=1.0) as pending:
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGINT)
                assert events == []
                assert pending.exit_code == 128 + signal.SIGTERM

    assert exited == []  # no escalation to a hard exit
    assert events == ["shutdown"]
    stderr = capfd.readouterr().err
    assert len(_marker_lines("signal-deferred", stderr)) == 1
    assert len(_marker_lines("signal-ignored", stderr)) == 2


def test_the_budget_is_absolute_from_the_signal_not_per_boundary():
    """A rank cannot accumulate a fresh budget at each rendezvous.

    The budget bounds how long a rank that is leaving waits for peers that may
    never arrive, measured from the local event, so time spent walking to a
    rendezvous is spent out of it.
    """
    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit):
            with defer_termination(budget=0.3) as pending:
                # nothing recorded yet, so no clock is running: a healthy rank
                # must never be the one holding a short deadline
                assert pending.seconds_remaining() is None
                signal.raise_signal(signal.SIGTERM)
                observed = []
                for _ in range(3):
                    time.sleep(0.01)
                    remaining = pending.seconds_remaining()
                    assert remaining is not None
                    observed.append(remaining)
                assert observed == sorted(observed, reverse=True)
                assert len(set(observed)) == 3
                assert max(observed) < 0.3


def test_a_peer_stop_tears_down_even_though_no_signal_arrived():
    """A rank that only ever read a peer's stop must still tear down.

    Without this it would leave the loop and walk into the next collective, whose
    peers have already left -- so it would hang there and be SIGKILLed with its
    communicators open.
    """
    events = []

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(SystemExit) as excinfo:
            with defer_termination(budget=1.0) as pending:
                pending.note_peer_stop()

    assert events == ["shutdown"]
    # a peer's stop carries no signal number, so SIGTERM's code by convention
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_an_exception_inside_a_deferral_tears_down_without_masking_it():
    """The exception path must tear down and still lose nothing of the raise.

    Today an exception skips `shutdown()` entirely and the rank exits with its
    communicators open. Tearing down here fixes that, but a `sys.exit` on the way
    out would replace the traceback with an exit code, so this path returns.
    """
    events = []
    add_post_shutdown_callback(lambda: events.append("callback"))

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        with pytest.raises(ValueError, match="a NaN in the loss"):
            with defer_termination(budget=1.0):
                raise ValueError("a NaN in the loss")

    assert events == ["shutdown", "callback"]


def test_a_deferral_that_never_stops_writes_one_diagnostic_line(capfd):
    """A rank that never reaches a stopping point must not go unrecorded.

    The timer only writes: the sole lever a timer thread has over a wedged main
    thread is `os._exit`, which is the fabric fault this module exists to avoid.
    """
    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit):
            with defer_termination(budget=0.05) as pending:
                signal.raise_signal(signal.SIGTERM)
                assert pending.requested
                time.sleep(0.3)  # past 2 x budget, without reaching a stop

    overrun = _marker_lines("deferral-overrun", capfd.readouterr().err)
    assert len(overrun) == 1
    assert "since=0s" in overrun[0]


def test_a_deferral_that_stops_promptly_writes_no_diagnostic_line(capfd):
    with handle_termination_signals(shutdown=lambda: None):
        with pytest.raises(SystemExit):
            with defer_termination(budget=0.05) as pending:
                signal.raise_signal(signal.SIGTERM)
                assert pending.requested

    time.sleep(0.3)  # long enough that an uncancelled timer would have fired
    assert _marker_lines("deferral-overrun", capfd.readouterr().err) == []


def test_deferral_is_inert_in_a_dataloader_worker(monkeypatch):
    """A worker must not swallow the signal that would have killed it.

    A fork-started worker inherits both the handler and the deferral registry, so
    a deferral that registered here would leave the worker alive and deaf: it
    would neither die as it would have without the inheritance nor gain anyone to
    poll what it recorded.
    """
    events = []
    pendings = []

    with handle_termination_signals(shutdown=lambda: events.append("shutdown")):
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
        with pytest.raises(SystemExit) as excinfo:
            with defer_termination(budget=1.0) as pending:
                pendings.append(pending)
                signal.raise_signal(signal.SIGTERM)

    assert pendings[0].requested is False
    assert events == ["shutdown"]
    assert excinfo.value.code == 128 + signal.SIGTERM


def test_a_deferral_off_the_main_thread_does_not_claim_the_registry():
    """A thread cannot own the disposition, so it must not own the registry.

    Nothing would ever record a signal into it, and claiming it would lock the
    main thread out of the one registration there is.
    """
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with defer_termination(budget=1.0):
                with defer_termination(budget=1.0):
                    pass
        except BaseException as err:
            errors.append(err)  # a thread's exception would otherwise be lost

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert errors == []


def test_nested_defer_termination_is_rejected():
    """The registry is process-global, so only one scope can own the teardown."""
    with defer_termination(budget=1.0):
        with pytest.raises(RuntimeError, match="Nested"):
            with defer_termination(budget=1.0):
                pass


def test_a_forked_child_in_a_deferral_does_not_record_intent():
    """The pid guard must run before the deferral branch.

    A forked worker inherits the registry along with the handler, so if the
    deferral branch ran first the worker would record intent and return -- left
    alive, deaf, and with nobody to poll it -- instead of dying as it would have
    without the inheritance.
    """
    read_fd, write_fd = os.pipe()

    def report(event: bytes) -> None:
        # the child cannot report back through the parent's objects
        os.write(write_fd, event)

    add_post_shutdown_callback(lambda: report(b"callback"))

    with handle_termination_signals(shutdown=lambda: report(b"shutdown")):
        with defer_termination(budget=1.0):
            pid = os.fork()
            if pid == 0:
                # never let a forked copy of the test session escape back into
                # pytest, whatever the handler does
                try:
                    os.close(read_fd)
                    signal.raise_signal(signal.SIGTERM)
                    report(b"deferred")
                except BaseException:
                    pass
                finally:
                    os._exit(0)

            os.close(write_fd)
            ready: list[int] = []
            try:
                ready, _, _ = select.select([read_fd], [], [], 30.0)
                observed = os.read(read_fd, 4096) if ready else b"child hung"
            finally:
                os.close(read_fd)
                if not ready:
                    os.kill(pid, signal.SIGKILL)
                _, status = os.waitpid(pid, 0)

    assert observed == b""
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGTERM


def test_marker_lines_carry_the_documented_fields(capfd, monkeypatch):
    """One line, one write, fields in a fixed order.

    The reader's whole recipe is `grep` over these lines, and a rank that failed
    is identified by the absence of its own, so every line has to carry the rank
    label and `installed_pid` -- which is what tells a real rank from a
    fork-started DataLoader worker sharing its parent's RANK.
    """
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")

    with handle_termination_signals(shutdown=lambda: None):
        write_marker("stop-agreed", batch="41200", world="8")
    write_marker("stop-agreed", batch="41201", world="8")

    installed, uninstalled = _marker_lines("stop-agreed", capfd.readouterr().err)
    fields = installed.split(" ")
    assert [field.split("=")[0] for field in fields[1:]] == [
        "rank",
        "local_rank",
        "pid",
        "installed_pid",
        "wall",
        "mono",
        "batch",
        "world",
    ]
    assert fields[0] == "fme-stop:stop-agreed"
    assert f"rank=3 local_rank=1 pid={os.getpid()} installed_pid={os.getpid()}" in (
        installed
    )
    assert "batch=41200 world=8" in installed
    # with no handler installed there is no pid to compare against
    assert "installed_pid=?" in uninstalled


def test_marker_lines_survive_an_unwritable_stderr(monkeypatch):
    """A marker is evidence about a failure and may not become one."""

    def refuse(fd, data):
        raise OSError("stderr is gone")

    monkeypatch.setattr(os, "write", refuse)
    write_marker("stop-agreed", batch="41200")


@pytest.mark.medium_duration
def test_the_watchdog_aborts_before_its_backstop_exit():
    """A rank that gives up must abort its communicators, not just vanish.

    An `os._exit` taken with communicators open is what drops this rank's NVLink
    peers abruptly. Aborting first at least makes the release ordered, and may
    release this rank's own main thread from the collective; the hard exit stays
    as a named backstop for when it does not.
    """
    result = _run_handler_program(
        """
        import signal, threading
        from fme.core.distributed import shutdown as shutdown_module
        from fme.core.distributed.shutdown import handle_termination_signals

        shutdown_module._ABORT_BACKSTOP = 0.3

        def hang():
            threading.Event().wait()

        def abort():
            print("abort ran", flush=True)

        with handle_termination_signals(
            shutdown=hang, teardown_timeout=1.0, abort=abort
        ):
            signal.raise_signal(signal.SIGTERM)
        """
    )

    assert result.returncode == 128 + signal.SIGTERM
    assert result.stdout.splitlines() == ["abort ran"]
    assert "fme-stop:watchdog-abort" in result.stderr
    assert "timeout=1" in result.stderr
    assert (
        "Distributed shutdown did not complete within 1s, aborting the local "
        "communicator." in result.stderr
    )


@pytest.mark.medium_duration
def test_a_released_main_thread_is_not_exited_mid_callback():
    """The backstop must stand down if the abort released the main thread.

    An abort that works releases the main thread at once, so it can cancel the
    watchdog while the watchdog thread is still between its abort and its
    backstop. `Timer.cancel()` on an unstarted timer does nothing, so without the
    cancelled flag the backstop would start unopposed and exit the process out
    from under the checkpoint write the released main thread had just begun.
    """
    result = _run_handler_program(
        """
        import signal, threading, time
        from fme.core.distributed import shutdown as shutdown_module
        from fme.core.distributed.shutdown import (
            add_post_shutdown_callback,
            handle_termination_signals,
        )

        shutdown_module._ABORT_BACKSTOP = 0.2

        released = threading.Event()

        def hang():
            released.wait()      # the abort below is what lets this return

        def abort():
            released.set()

        def slow_write():
            time.sleep(1.0)      # stands in for the multi-GB torch.save
            print("callback complete", flush=True)

        add_post_shutdown_callback(slow_write)
        try:
            with handle_termination_signals(
                shutdown=hang, teardown_timeout=0.5, abort=abort
            ):
                signal.raise_signal(signal.SIGTERM)
        except SystemExit as err:
            # the graceful exit and the backstop's os._exit share an exit code,
            # so the code cannot be the evidence
            print(f"sys.exit {err.code}", flush=True)
            raise
        """
    )

    assert result.stdout.splitlines() == [
        "callback complete",
        f"sys.exit {128 + signal.SIGTERM}",
    ]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_a_signal_outside_a_deferral_exits_128_plus_signum():
    """The immediate path is unchanged at process level, not only in process.

    pytest catches the graceful `SystemExit`, so the exit code a real rank hands
    the launcher can only be observed from outside.
    """
    result = _run_handler_program(
        """
        import signal
        from fme.core.distributed.shutdown import (
            defer_termination,
            handle_termination_signals,
        )

        with handle_termination_signals(
            shutdown=lambda: print("shutdown", flush=True)
        ):
            with defer_termination(budget=1.0):
                pass
            signal.raise_signal(signal.SIGTERM)
        """
    )

    assert result.returncode == 128 + signal.SIGTERM
    assert result.stdout.splitlines() == ["shutdown"]
    assert "fme-stop:shutdown-returned" in result.stderr
