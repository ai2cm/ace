import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch.utils.data

from fme.core.distributed import shutdown
from fme.core.distributed.shutdown import (
    TERMINATION_SIGNALS,
    abort_and_exit_on_termination,
    park_if_terminating,
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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.1,
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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        def signal_later():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=signal_later, daemon=True).start()
        lock = threading.Lock()
        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.1,
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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        def abort():
            raise RuntimeError("no process group")

        with abort_and_exit_on_termination(
            abort=abort, grace_period=0.1, park_deadline=0.1
        ):
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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        with abort_and_exit_on_termination(
            abort=lambda: None, grace_period={grace_period}, park_deadline=0.1
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
def test_post_abort_callbacks_run_after_the_abort_and_cannot_stop_the_exit():
    """Callbacks (the Trainer's restart checkpoint) run once the abort has
    released the rank's own kernels, and a raising one must not block the
    exit or its successors."""
    result = _run_listener_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import (
            add_post_abort_callback,
            abort_and_exit_on_termination,
        )

        released = threading.Event()

        def abort():
            print("abort", flush=True)
            released.set()

        def broken():
            raise RuntimeError("no checkpoint for you")

        add_post_abort_callback(lambda: print("first", flush=True))
        add_post_abort_callback(broken)
        add_post_abort_callback(lambda: print("second", flush=True))
        with abort_and_exit_on_termination(
            abort=abort, grace_period=0.1, park_deadline=0.1
        ):
            signal.raise_signal(signal.SIGTERM)
            released.wait()  # blocked "in the collective" until the abort
            raise RuntimeError("rank unwinding after its collective died")
        """
    )

    assert result.stdout.split() == ["abort", "first", "second"]
    assert "no checkpoint for you" in result.stderr
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_callbacks_wait_for_the_main_thread_to_stop_running():
    """The abort releases only a main thread blocked in a collective; one
    between collectives keeps computing until the next collective raises. A
    callback that reads training state (the restart checkpoint) must not run
    until the main thread has blocked in the context's exit, or it would
    snapshot state mid-mutation.
    """
    result = _run_listener_program(
        """
        import signal, threading, time
        from fme.core.distributed.shutdown import (
            add_post_abort_callback,
            abort_and_exit_on_termination,
        )

        released = threading.Event()

        def abort():
            print("abort", flush=True)
            released.set()

        add_post_abort_callback(lambda: print("callback", flush=True))
        with abort_and_exit_on_termination(
            abort=abort, grace_period=0.1, park_deadline=0.1
        ):
            signal.raise_signal(signal.SIGTERM)
            released.wait()
            time.sleep(1.0)  # compute continuing between collectives
            print("still training", flush=True)
            raise RuntimeError("the next collective died")
        """
    )

    assert result.stdout.splitlines() == ["abort", "still training", "callback"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_callbacks_are_skipped_when_the_main_thread_never_stops():
    """A main thread that never unwinds (stuck outside any collective) keeps
    mutating whatever it is stuck in; a snapshot taken alongside it could be
    torn, so no snapshot is taken -- but the grace period and exit still
    happen."""
    result = _run_listener_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import (
            add_post_abort_callback,
            abort_and_exit_on_termination,
        )

        add_post_abort_callback(lambda: print("callback", flush=True))
        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            state_freeze_timeout=0.2,
            park_deadline=0.1,
        ):
            signal.raise_signal(signal.SIGTERM)
            threading.Event().wait()  # never unwinds
        """
    )

    assert result.stdout.split() == ["abort"]
    assert "skipping post-abort callbacks" in result.stderr
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_a_dead_stderr_cannot_stop_the_abort_or_the_exit():
    """``os.write`` to a pipe whose reader is gone raises EPIPE (Python
    ignores SIGPIPE), and mid-preemption the log collector may die before the
    ranks do. Losing the listener's messages must not lose the abort, the
    grace period, or the exit code."""
    result = _run_listener_program(
        """
        import os, signal, threading
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        r, w = os.pipe()
        os.close(r)
        os.dup2(w, 2)  # every stderr write now raises BrokenPipeError
        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.1,
        ):
            signal.raise_signal(signal.SIGTERM)
            threading.Event().wait()
        """
    )

    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        released = threading.Event()

        def abort():
            print("abort", flush=True)
            released.set()

        with abort_and_exit_on_termination(
            abort=abort, grace_period=1.0, park_deadline=0.1
        ):
            signal.raise_signal(signal.SIGTERM)
            released.wait()  # blocked "in the collective" until the abort
            raise RuntimeError("rank unwinding after its collective died")
        """
    )

    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_repeated_signal_cannot_cut_the_grace_period_short(sig):
    """A second signal is routine whenever a human is involved: Ctrl-C hits
    every rank via the tty and again via torchrun's forwarding, and an
    impatient operator repeats it. If the context's exit restored the previous
    dispositions (typically ``SIG_DFL``) while the listener was still waiting
    out the grace period, that second signal would kill the rank instantly --
    before its peers' aborts had finished.
    """
    grace_period = 2.0
    program = textwrap.dedent(
        f"""
        import threading
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        released = threading.Event()

        def abort():
            print("abort", flush=True)
            released.set()

        try:
            with abort_and_exit_on_termination(
                abort=abort, grace_period={grace_period}, park_deadline=0.1
            ):
                print("ready", flush=True)
                released.wait()  # blocked "in the collective" until the abort
                raise RuntimeError("rank unwinding after its collective died")
        except BaseException as exc:  # noqa: B036
            print(f"escaped the context: {{type(exc).__name__}}", flush=True)
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
        proc.send_signal(sig)
        # once the abort has run, the released main thread is unwinding into
        # the context's exit; give it time to get there before the repeat
        assert proc.stdout.readline().strip() == "abort"
        time.sleep(0.5)
        proc.send_signal(sig)
        proc.wait(timeout=30)
        elapsed = time.monotonic() - start
        remaining_stdout = proc.stdout.read()
    finally:
        proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    assert "escaped the context" not in remaining_stdout
    assert proc.returncode == 128 + sig
    assert elapsed >= grace_period * 0.9


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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.1,
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
def test_second_generation_fork_does_not_close_recycled_fds():
    """A fork-started worker's own fork (dataset code using multiprocessing)
    runs the at-fork disarm a second time. The first disarm closed the
    inherited pipe fds, so it must also clear the module state -- or the
    second run closes fd numbers the worker has since reused for something
    else.
    """
    result = _run_listener_program(
        """
        import os
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        with abort_and_exit_on_termination(abort=lambda: None, grace_period=0.1):
            pid = os.fork()
            if pid == 0:  # worker: its at-fork disarm closed the pipe fds
                a, b = os.pipe()  # recycle those fd numbers
                gpid = os.fork()  # grandchild: the at-fork hook runs again
                if gpid == 0:
                    try:
                        os.fstat(a)
                        os.fstat(b)
                    except OSError:
                        os._exit(1)
                    os._exit(0)
                _, status = os.waitpid(gpid, 0)
                os._exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 2)
            _, status = os.waitpid(pid, 0)
            intact = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            print(f"recycled fds intact: {intact}", flush=True)
        """
    )

    assert result.stdout.splitlines() == ["recycled fds intact: True"]
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
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        signal.signal(signal.SIGCHLD, lambda signum, frame: None)
        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.1,
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

    with abort_and_exit_on_termination(abort=lambda: None):
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
        with abort_and_exit_on_termination(abort=lambda: None):
            observed.append(signal.getsignal(signal.SIGTERM))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert observed == [original]


def test_no_listener_installed_in_a_dataloader_worker(monkeypatch):
    """The DataLoader owns its workers' lifecycle, so leave their signals alone."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    original = signal.getsignal(signal.SIGTERM)

    with abort_and_exit_on_termination(abort=lambda: None):
        assert signal.getsignal(signal.SIGTERM) is original


@pytest.mark.medium_duration
def test_a_parked_main_thread_is_aborted_without_waiting_for_the_deadline():
    """Parking at a loop boundary tells the listener the rank's collectives
    are idle: the abort proceeds after the settle period rather than waiting
    out the park deadline -- the deadline here dwarfs the observed runtime,
    so waiting it out would fail the timing bound -- and the post-abort
    callbacks run against the boundary-frozen state rather than being
    skipped.
    """
    start = time.monotonic()
    result = _run_listener_program(
        """
        import signal
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            add_post_abort_callback,
            park_if_terminating,
        )

        add_post_abort_callback(lambda: print("callback", flush=True))
        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=30.0,
            settle_period=0.05,
        ):
            signal.raise_signal(signal.SIGTERM)
            while True:  # the training loop: a boundary between batches
                park_if_terminating()
        """
    )
    elapsed = time.monotonic() - start

    assert "parked at a loop boundary" in result.stderr
    assert "has not parked" not in result.stderr
    assert result.stdout.split() == ["abort", "callback"]
    assert result.returncode == 128 + signal.SIGTERM
    assert elapsed < 25.0  # interpreter startup dominates; the deadline would add 30


@pytest.mark.medium_duration
def test_the_settle_period_holds_the_abort_after_the_park():
    """A parked rank's own collectives are complete, but a peer's kernel for
    the last of them may still be retiring -- actively reading this rank's
    memory. The settle period between the park and the abort is what lets it
    drain, so the abort must not come earlier.
    """
    settle_period = 0.5
    result = _run_listener_program(
        f"""
        import signal, time
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            park_if_terminating,
        )

        boundary = {{"at": 0.0}}

        def abort():
            print(f"settle {{time.monotonic() - boundary['at']:.3f}}", flush=True)

        with abort_and_exit_on_termination(
            abort=abort,
            grace_period=0.1,
            park_deadline=5.0,
            settle_period={settle_period},
        ):
            signal.raise_signal(signal.SIGTERM)
            while True:
                boundary["at"] = time.monotonic()
                park_if_terminating()
        """
    )

    label, observed = result.stdout.split()
    assert label == "settle"
    assert float(observed) >= settle_period * 0.9
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_the_deadline_bounds_the_wait_for_a_main_thread_that_never_parks():
    """A main thread wedged in a collective a peer will never complete cannot
    park; its kernels are waiting rather than transferring, so after the
    deadline the abort proceeds without it.
    """
    result = _run_listener_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import abort_and_exit_on_termination

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.2,
        ):
            signal.raise_signal(signal.SIGTERM)
            threading.Event().wait()  # wedged "in a collective"
        """
    )

    assert "has not parked" in result.stderr
    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_only_the_main_thread_parks():
    """Any thread may call the park check, but only the main thread's arrival
    at a loop boundary says anything about the rank's collectives -- a side
    thread (a DataLoader pin-memory thread, a metrics thread) must return
    immediately rather than freeze training state the main thread is still
    mutating.
    """
    result = _run_listener_program(
        """
        import signal, threading, time
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            park_if_terminating,
        )

        def side_thread():
            park_if_terminating()
            print("side thread returned", flush=True)

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=2.0,
        ):
            signal.raise_signal(signal.SIGTERM)
            time.sleep(0.1)  # let the listener take ownership of the exit
            threading.Thread(target=side_thread).start()
            threading.Event().wait()  # the main thread itself never parks
        """
    )

    assert "side thread returned" in result.stdout
    assert "has not parked" in result.stderr
    assert result.returncode == 128 + signal.SIGTERM


def test_park_is_a_noop_without_a_pending_termination():
    park_if_terminating()  # no listener context armed at all
    with abort_and_exit_on_termination(abort=lambda: None):
        park_if_terminating()  # armed, but no signal has arrived


@pytest.mark.medium_duration
def test_the_drain_runs_before_the_park_and_the_abort():
    """The loop reaching a boundary is a host-side fact; the batch's last
    collectives may still be running on the device. The park must drain them
    before freezing state, and the abort must come after the drain.
    """
    result = _run_listener_program(
        """
        import signal
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            park_if_terminating,
        )

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=5.0,
            settle_period=0.05,
            drain=lambda: print("drain", flush=True),
        ):
            signal.raise_signal(signal.SIGTERM)
            while True:
                park_if_terminating()
        """
    )

    assert result.stdout.split() == ["drain", "abort"]
    assert "parked at a loop boundary" in result.stderr
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_a_drain_that_hangs_leaves_the_deadline_path_in_charge():
    """A drain wedged on a collective a peer will never complete must not
    freeze training state -- the listener's deadline abort assumes nothing
    about this rank's kernels, and is what releases the situation.
    """
    result = _run_listener_program(
        """
        import signal, threading
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            park_if_terminating,
        )

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.3,
            drain=lambda: threading.Event().wait(),
        ):
            signal.raise_signal(signal.SIGTERM)
            while True:
                park_if_terminating()
        """
    )

    assert "draining local device work" in result.stderr
    assert "parked at a loop boundary" not in result.stderr
    assert "has not parked" in result.stderr
    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_a_drain_that_raises_forfeits_the_callbacks_but_not_the_exit():
    """A drain that raises (a sticky device error) means the kernel state is
    unknown: training state must not be frozen -- so the callbacks are
    skipped rather than snapshot beside it -- and the deadline abort and exit
    must still happen.
    """
    result = _run_listener_program(
        """
        import signal
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            add_post_abort_callback,
            park_if_terminating,
        )

        def drain():
            raise RuntimeError("device poisoned")

        add_post_abort_callback(lambda: print("callback", flush=True))
        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=0.3,
            state_freeze_timeout=0.2,
            drain=drain,
        ):
            signal.raise_signal(signal.SIGTERM)
            while True:
                park_if_terminating()
        """
    )

    assert "Draining local device work failed" in result.stderr
    assert "device poisoned" in result.stderr
    assert "skipping post-abort callbacks" in result.stderr
    assert result.stdout.split() == ["abort"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_a_child_forked_after_the_signal_does_not_park():
    """``os.fork`` copies a set ``listener_owns_exit`` into the child, whose
    surviving thread is re-designated the main thread: without the at-fork
    clear of the armed context, the child's next park check would block it
    forever on the parent's pending termination.
    """
    result = _run_listener_program(
        """
        import os, signal, time
        from fme.core.distributed.shutdown import (
            abort_and_exit_on_termination,
            park_if_terminating,
        )

        with abort_and_exit_on_termination(
            abort=lambda: print("abort", flush=True),
            grace_period=0.1,
            park_deadline=5.0,
            settle_period=0.05,
        ):
            signal.raise_signal(signal.SIGTERM)
            time.sleep(0.2)  # let the listener take ownership of the exit
            pid = os.fork()
            if pid == 0:
                park_if_terminating()  # must return, not park
                os._exit(7)
            _, status = os.waitpid(pid, 0)
            print(f"child exit: {os.WEXITSTATUS(status)}", flush=True)
            park_if_terminating()  # the parent itself parks here
        """
    )

    assert "child exit: 7" in result.stdout
    assert "parked at a loop boundary" in result.stderr
    assert result.returncode == 128 + signal.SIGTERM


def test_park_returns_immediately_in_a_dataloader_worker(monkeypatch):
    """A worker's loop position says nothing about the parent's collectives,
    and a parked worker would hang the DataLoader that owns it, so the guard
    must return even with a termination pending.
    """
    coordination = shutdown._ExitCoordination()
    coordination.mark_listener_owns_exit()
    monkeypatch.setattr(
        shutdown,
        "_armed_context",
        shutdown._ArmedContext(coordination, lambda: None),
    )
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())

    park_if_terminating()  # returning at all is the assertion
