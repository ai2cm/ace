"""Tier 3: the exit paths, in subprocesses the test spawns.

None of these can run in-session. `make test_parallel` is one pytest session per
rank, so a rank that exits takes its whole session down without a summary and
torchrun kills its peers, failing the entire ``-m parallel`` collection for that
configuration. And `Distributed.context()` is already entered for the session and
refuses to nest, while these tests are about what a real entrypoint does inside
it.

**They are not cheap.** Seven torchrun launches, each ``@pytest.mark.serial``, so
each holds the xdist write lock for the whole of a ``make test`` run at roughly
8-15s apiece. That is the price of asserting a process exit code at all; nothing
cheaper can.

So each writes a driver to ``tmp_path`` and launches it under ``torchrun``, the
way `test_shutdown_dataloader.py` does: ``start_new_session=True`` as torchrun
launches a rank, and ``os.killpg`` where the claim is about a scheduler signalling
a container. ``FME_FORCE_CPU=1`` throughout, so gloo, so they run on the CPU job
too. Every timeout is injected small: no driver waits out a real agreement budget
or a real 20s teardown.

Ranks' stderr is inherited rather than redirected, so the ``fme-stop:`` marker
lines land in the launcher's own stderr and are what the assertions read. Exit
codes come from files the drivers write, because torchrun reports its own status
rather than each rank's -- except on the paths that end in `os._exit`, where no
``SystemExit`` is raised for a driver to catch and the ``hard-exit`` marker is what
carries the code.

**Where the claim is a bound on the exit, the exit is observed from outside.** A
timestamp a driver writes before leaving measures the *call*, and the call is not
what can block: an abandoned gloo group makes interpreter finalization wait for the
wedged peer to die -- see
`fme.core.distributed.shutdown.PendingStop.require_hard_exit` for the measurement.
So `_wait_for_death` watches ``/proc`` for the rank's own pid, and the wedged rank
holds its socket open until the launcher says the rank under test is gone --
otherwise the peer's exit unblocks the very wait being measured.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fme.core.distributed.cooperative_stop import _MIN_DEADLINE

# torchrun's own environment would send the driver's ranks down a rendezvous that
# is not theirs; the launcher below is the one that sets these.
_RANK_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "SLURM_NTASKS",
)

# Sized so that the module's *own* diagnostics are what a maintainer sees on a
# hang, which is the case where a diagnostic is worth most. The figure that has to
# hold is the sum along the longest path of any one test, against the suite's
# autouse 90s SIGALRM: two readiness waits, one death wait and the launch, i.e.
# 15 + 15 + 10 + 25 = 65s, leaving 25s of the alarm unspent. Process startup is
# *inside* the first readiness wait rather than additional to it, since nothing is
# waited on before the launch. Observed runtime is ~8-15s per test, so each figure
# is still at least twice the real one.
_LAUNCH_TIMEOUT = 25.0
_READY_TIMEOUT = 15.0
_DEATH_TIMEOUT = 10.0

_SIGTERM_EXIT_CODE = 128 + int(signal.SIGTERM)

# torchrun puts the *script's* directory on `sys.path`, not the working directory,
# so a driver written to `tmp_path` would import whichever `fme` happens to be
# installed rather than the one under test.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _driver_env(**overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _RANK_ENV}
    env.pop("FME_DISTRIBUTED_BACKEND", None)
    env.pop("FME_DISTRIBUTED_H", None)
    env.pop("FME_DISTRIBUTED_W", None)
    env["FME_FORCE_CPU"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    env.update(overrides)
    return env


def _launch(
    source: str,
    tmp_path: Path,
    *,
    nproc: int = 2,
    env: dict[str, str] | None = None,
) -> "subprocess.Popen[str]":
    """Write a driver and launch it under torchrun in its own session."""
    script = tmp_path / "driver.py"
    script.write_text(textwrap.dedent(source))
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            str(nproc),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            "127.0.0.1:0",
            str(script),
            str(tmp_path),
        ],
        start_new_session=True,  # as torchrun launches a rank
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_driver_env() if env is None else env,
    )


def _wait_for_file(path: Path, child: "subprocess.Popen[str]") -> None:
    give_up_at = time.monotonic() + _READY_TIMEOUT
    while not path.exists():
        if child.poll() is not None:
            out, err = child.communicate()
            raise AssertionError(
                f"the driver exited with {child.returncode} before writing "
                f"{path.name}\nstdout:\n{out}\nstderr:\n{err}"
            )
        if time.monotonic() > give_up_at:
            raise AssertionError(f"timed out waiting for {path.name}")
        time.sleep(0.05)


def _running(pid: int) -> bool:
    """Whether ``pid`` is still a live process, a zombie counting as dead.

    The kernel sets ``Z`` when the process exits, and the parent that will reap it is
    torchrun rather than this process, so waiting for the entry to disappear would
    time a reap rather than a death.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    # the executable name is in parentheses and may itself contain spaces, so the
    # state field is read after the last ``)`` rather than by splitting the line
    fields = stat.rpartition(")")[2].split()
    return bool(fields) and fields[0] != "Z"


def _wait_for_death(pid: int, tmp_path: Path) -> float:
    """When ``pid`` stopped running, observed from outside it.

    The thing under test on the abandonment paths is how long the rank takes to
    *die*, which nothing the rank writes can report: a timestamp written before the
    exit measures the call, and the call is precisely what can block for as long as
    a wedged peer lives.

    Writes ``dead`` on the way out, whether or not the rank died in time, because a
    wedged peer is waiting on that file to leave -- it must not close its socket
    before this, since doing so would unblock the wait being measured.
    """
    give_up_at = time.monotonic() + _DEATH_TIMEOUT
    try:
        while _running(pid):
            if time.monotonic() > give_up_at:
                raise AssertionError(
                    f"the rank under test (pid {pid}) was still running "
                    f"{_DEATH_TIMEOUT:.0f}s after it began leaving, while its peer "
                    "was still holding its socket open"
                )
            time.sleep(0.02)
        return time.monotonic()
    finally:
        (tmp_path / "dead").write_text("go")


def _finish(child: "subprocess.Popen[str]", tmp_path: Path) -> str:
    """Wait for the launch to end and return its stderr, killing it if it hangs."""
    pgid = os.getpgid(child.pid)
    try:
        _, err = child.communicate(timeout=_LAUNCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        _, err = child.communicate()
        raise AssertionError(f"the launch never finished; stderr:\n{err}")
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return err


def _marker_fields(event: str, captured: str, field: str) -> list[str]:
    """Every value of ``field`` on the ``event`` lines, one per emitting rank."""
    prefix = f"fme-stop:{event} "
    values = []
    for line in captured.splitlines():
        if not line.startswith(prefix):
            continue
        for part in line.split():
            name, _, value = part.partition("=")
            if name == field:
                values.append(value)
    return values


def _exit_codes(tmp_path: Path, nproc: int) -> dict[str, str]:
    return {
        str(rank): (tmp_path / f"exit.{rank}").read_text()
        for rank in range(nproc)
        if (tmp_path / f"exit.{rank}").exists()
    }


# Recording the exit code from inside the driver, because torchrun reports its own
# status rather than each rank's. Only for the drivers whose exit is a `SystemExit`:
# on the abandonment paths the exit is an `os._exit`, nothing inside the interpreter
# sees it coming, and the ``hard-exit`` marker is what carries the code instead.
_RECORD_EXIT = """
        def record_exit(code):
            (OUT / f"exit.{os.environ['RANK']}").write_text(str(code))
"""


@pytest.mark.slow
@pytest.mark.medium_duration
@pytest.mark.serial
def test_both_ranks_exit_at_the_same_batch_with_the_backend_released(tmp_path):
    """The whole thesis, end to end, and the evidence channel with it.

    Two ranks under torchrun with a real per-iteration all-reduce on the default
    group, signalled the way a scheduler signals a container: one ``killpg`` to
    the whole process group. Both must leave the loop at the *same* batch, with
    their backends released and their post-shutdown callbacks run -- which is
    exactly what a reader has to be able to confirm from a container log, and
    exactly what today's code cannot promise.
    """
    child = _launch(
        f"""
        import os, sys, time
        from pathlib import Path

        import torch
        import torch.distributed

        from fme.core.distributed import (
            Distributed,
            add_post_shutdown_callback,
            cooperative_stop,
        )

        OUT = Path(sys.argv[1])
        {_RECORD_EXIT}
        rank = os.environ["RANK"]
        add_post_shutdown_callback(
            lambda: (OUT / f"callback.{{rank}}").write_text("ran")
        )
        try:
            with Distributed.context():
                with cooperative_stop(budget=1.0) as stop:
                    for index in range(1000):
                        torch.distributed.all_reduce(torch.ones(1))
                        time.sleep(0.05)
                        if index == 2:
                            (OUT / f"ready.{{rank}}").write_text("go")
                        if stop.agreed(index):
                            break
        except SystemExit as err:
            record_exit(err.code)
            raise
        """,
        tmp_path,
    )
    try:
        _wait_for_file(tmp_path / "ready.0", child)
        _wait_for_file(tmp_path / "ready.1", child)
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)  # as the scheduler does
    except BaseException:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        raise
    stderr = _finish(child, tmp_path)

    batches = _marker_fields("stop-agreed", stderr, "batch")
    assert len(batches) == 2, stderr
    assert len(set(batches)) == 1, f"the ranks stopped at different batches: {batches}"
    assert len(_marker_fields("shutdown-returned", stderr, "elapsed")) == 2, stderr
    assert "did not complete" not in stderr
    assert "fme-stop:agreement-timeout" not in stderr
    assert "fme-stop:agreement-abandoned" not in stderr
    assert "fme-stop:index-mismatch" not in stderr
    assert _exit_codes(tmp_path, 2) == {
        "0": str(_SIGTERM_EXIT_CODE),
        "1": str(_SIGTERM_EXIT_CODE),
    }
    assert (tmp_path / "callback.0").exists() and (tmp_path / "callback.1").exists()


@pytest.mark.slow
@pytest.mark.medium_duration
@pytest.mark.serial
def test_a_wedged_rank_does_not_hold_a_healthy_rank_past_its_budget(tmp_path):
    """A peer that never reaches the boundary does not hold the others there.

    Rank 1 blocks SIGTERM with ``pthread_sigmask`` and parks, which is a
    deterministic stand-in for "its main thread is inside a C-level call so its
    handler cannot run" and needs no real NCCL wedge.

    The two assertions that matter beyond the bound are that rank 0's teardown
    *returned* and its callbacks *ran*. Both are true only because the abandoned
    agreement group is left behind rather than reclaimed: reclaiming it would
    block in the gloo destructor without bound, the phase would never reach the
    callbacks, and the restart checkpoint would be lost by construction on exactly
    the path where it is most valuable.

    Rank 0 raises its own SIGTERM rather than waiting for one from outside, so
    that its budget is provably armed before it enters the exchange rank 1 will
    never join. A signal from outside landing a moment later would find rank 0
    already inside that exchange carrying the default group's own timeout, which is
    a different case with a different bound; the test named
    ``test_both_ranks_exit_at_the_same_batch_with_the_backend_released`` in this
    module is where an external signal is the claim.

    **What is timed is rank 0's death, not its call to exit**, and rank 1 holds its
    socket open until that death: an abandoned gloo group is disposed of by
    ``~ProcessGroupGloo()`` during interpreter finalization, which joins the worker
    thread still holding the abandoned operation and so ends only when the peer's
    socket closes. Timing a timestamp written before the exit, against a peer that
    had already gone, is a test that cannot observe the thing it appears to bound.

    Which of the two assertions does the work is worth being exact about, because
    the timing one does not: whether finalization joins that thread or leaks it is
    not deterministic, and the figures are in
    `fme.core.distributed.shutdown.PendingStop.require_hard_exit`, which states them
    once. This test's shape is the one that leaked, so the bound below passes either
    way on this machine today and the ``hard-exit`` marker is what pins the exit
    actually being taken hard. The bound is the backstop for the machine where it
    joins.
    """
    child = _launch(
        f"""
        import os, signal, sys, time
        from pathlib import Path

        import torch
        import torch.distributed

        from fme.core.distributed import (
            Distributed,
            add_post_shutdown_callback,
            cooperative_stop,
        )

        OUT = Path(sys.argv[1])
        rank = int(os.environ["RANK"])
        (OUT / f"pid.{{rank}}").write_text(str(os.getpid()))
        wedge_at = 2
        add_post_shutdown_callback(
            lambda: (OUT / f"callback.{{rank}}").write_text("ran")
        )
        with Distributed.context():
            with cooperative_stop(budget=1.0) as stop:
                for index in range(1000):
                    torch.distributed.all_reduce(torch.ones(1))
                    time.sleep(0.02)
                    if index == wedge_at:
                        if rank == 1:
                            # the handler can never run, so this rank never
                            # reaches the boundary below
                            signal.pthread_sigmask(
                                signal.SIG_BLOCK, {{signal.SIGTERM}}
                            )
                            # and its socket stays open until rank 0 is gone,
                            # so rank 0's exit is bounded by rank 0 alone
                            give_up_at = time.monotonic() + 20.0
                            while not (OUT / "dead").exists():
                                if time.monotonic() > give_up_at:
                                    break
                                time.sleep(0.02)
                            # a stand-in for the SIGKILL a real wedged rank
                            # eventually gets; nothing is asserted about it
                            os._exit(0)
                        signal.raise_signal(signal.SIGTERM)
                        (OUT / "started").write_text(str(time.monotonic()))
                    if stop.agreed(index):
                        break
        """,
        tmp_path,
    )
    try:
        _wait_for_file(tmp_path / "pid.0", child)
        _wait_for_file(tmp_path / "started", child)
        died = _wait_for_death(int((tmp_path / "pid.0").read_text()), tmp_path)
    except BaseException:
        (tmp_path / "dead").write_text("go")
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        raise
    stderr = _finish(child, tmp_path)

    started = float((tmp_path / "started").read_text())
    # The deadline that binds, plus the teardown and the callbacks, with slack. The
    # driver's 1s budget is below `_MIN_DEADLINE` and so is floored up to it, which
    # is the figure to compare against; the point is that it is bounded at all, and
    # that it is nothing like the time rank 1 remains alive.
    assert died - started < _MIN_DEADLINE + 4.0, died - started
    # no `SystemExit` reaches the driver on this path -- the exit is hard, because
    # the group left behind is what makes finalization wait for rank 1 -- so the
    # code comes off the marker the hard exit writes
    assert _marker_fields("hard-exit", stderr, "code") == [
        str(_SIGTERM_EXIT_CODE)
    ], stderr
    assert _marker_fields("agreement-timeout", stderr, "batch") == ["2"], stderr
    assert _marker_fields("agreement-abandoned", stderr, "batch") == ["2"], stderr
    assert len(_marker_fields("shutdown-returned", stderr, "elapsed")) == 1, stderr
    assert "did not complete" not in stderr
    assert (tmp_path / "callback.0").exists(), "the restart checkpoint was lost"


@pytest.mark.slow
@pytest.mark.medium_duration
@pytest.mark.serial
def test_a_rank_that_gives_up_with_no_local_event_exits(tmp_path):
    """The give-up path must end in an exit even with nothing recorded locally.

    No signal is sent anywhere in this test. Rank 1 simply stops reaching the
    boundary and rank 0 gives up waiting for it, holding nothing on its
    `PendingStop` -- no signal, no exception -- which is the state every rank the
    scheduler's signal did not reach is in. The exit, observed from outside the
    interpreter, is the assertion: a rank that merely broke out of the loop would
    run on into `alternate_shuffle` and the train-evaluation pass, hang there
    against peers that had gone, and be SIGKILLed with its communicators open.

    Rank 1 stays alive with its socket open until rank 0 is observed dead, for two
    reasons: so that rank 0 sees a deadline expiry rather than a closed socket, and
    so that rank 0's death is bounded by rank 0's own exit rather than by rank 1's.

    The no-local-event deadline is patched down from the default group's own
    timeout, which is tens of minutes; nothing else about the path is altered, and
    the deadline's *value* is what the tier-1 tests cover.
    """
    child = _launch(
        """
        import importlib, os, sys, time
        from pathlib import Path

        import torch
        import torch.distributed

        from fme.core.distributed import (
            Distributed,
            add_post_shutdown_callback,
            cooperative_stop,
        )

        # `import ... as` would bind the re-exported *function* of the same name,
        # and setting an attribute on that silently patches nothing
        module = importlib.import_module("fme.core.distributed.cooperative_stop")
        # tens of minutes otherwise, which is the whole point of that deadline
        module.no_local_event_deadline = lambda: 2.0

        OUT = Path(sys.argv[1])
        rank = int(os.environ["RANK"])
        (OUT / f"pid.{rank}").write_text(str(os.getpid()))
        leave_at = 2
        add_post_shutdown_callback(
            lambda: (OUT / f"callback.{rank}").write_text("ran")
        )
        with Distributed.context():
            with cooperative_stop() as stop:
                for index in range(1000):
                    torch.distributed.all_reduce(torch.ones(1))
                    time.sleep(0.02)
                    if index == leave_at and rank == 1:
                        (OUT / "left").write_text(str(time.monotonic()))
                        give_up_at = time.monotonic() + 20.0
                        while not (OUT / "dead").exists():
                            if time.monotonic() > give_up_at:
                                break
                            time.sleep(0.02)
                        os._exit(0)
                    if stop.agreed(index):
                        break
                (OUT / f"loop-exited.{rank}").write_text("yes")
            (OUT / f"scope-exited.{rank}").write_text("yes")
            torch.distributed.all_reduce(torch.ones(1))  # never reached
            (OUT / f"kept-going.{rank}").write_text("yes")
        """,
        tmp_path,
    )
    try:
        _wait_for_file(tmp_path / "pid.0", child)
        _wait_for_file(tmp_path / "left", child)
        died = _wait_for_death(int((tmp_path / "pid.0").read_text()), tmp_path)
    except BaseException:
        (tmp_path / "dead").write_text("go")
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        raise
    stderr = _finish(child, tmp_path)

    # the patched 2s deadline, plus the teardown and the callbacks, with slack --
    # and measured against a peer that was still alive, so it is rank 0's own exit
    # that is bounded
    left = float((tmp_path / "left").read_text())
    assert died - left < 6.0, died - left
    # no signal was sent, so `128 + SIGTERM` here is the peer-stop convention for a
    # stop that carries no signal number of its own. It comes off the hard exit's
    # marker because no `SystemExit` is raised on this path: the group rank 0 left
    # behind is what makes finalization wait for rank 1.
    assert _marker_fields("hard-exit", stderr, "code") == [
        str(_SIGTERM_EXIT_CODE)
    ], stderr
    assert not (
        tmp_path / "kept-going.0"
    ).exists(), "rank 0 left the loop and executed a further collective"
    assert not (
        tmp_path / "scope-exited.0"
    ).exists(), "leaving the cooperative scope did not dispatch the teardown"
    assert (tmp_path / "loop-exited.0").exists(), stderr
    assert _marker_fields("agreement-timeout", stderr, "batch") == ["2"], stderr
    assert _marker_fields("agreement-abandoned", stderr, "batch") == ["2"], stderr
    assert len(_marker_fields("shutdown-returned", stderr, "elapsed")) == 1, stderr
    # No signal is sent by this test. Rank 1 may still record one -- the agent
    # SIGTERMs it once rank 0 exits non-zero -- but rank 0, the rank under test,
    # must have given up with nothing of its own recorded, which is the whole point.
    assert "0" not in _marker_fields("signal-deferred", stderr, "rank"), stderr
    # and with no local event it was never bounded by a budget, so it claims no
    # bound of its own
    assert "fme-stop:agreement-bound" not in stderr, stderr
    assert (tmp_path / "callback.0").exists(), "the restart checkpoint was lost"


@pytest.mark.slow
@pytest.mark.medium_duration
@pytest.mark.serial
def test_a_raising_rank_that_abandons_an_exchange_dies_promptly_with_its_traceback(
    tmp_path,
):
    """The exception rider owes the same hard exit as every other abandonment path.

    Rank 0 raises inside the loop body with no signal anywhere, so its exit exchange
    is the one its peers cannot answer -- rank 1 has parked before reaching that
    boundary -- and the exchange is abandoned. Before this the branch returned with
    ``exit_process=False`` to keep the traceback, which left the rank to block in
    ``~ProcessGroupGloo()`` during finalization for as long as rank 1 lived and be
    SIGKILLed there: the launcher would read a signal death rather than a failure,
    and the rank would hold its GPU allocation throughout.

    So all three claims are asserted together, because dropping any one of them is
    how the fix regresses: the traceback still reaches stderr, the exit code is the
    exception's, and the death is bounded by rank 0's own exchange rather than by
    rank 1's demise -- rank 1 keeps its socket open until the launcher has seen rank
    0 go, so the bound is genuinely rank 0's.
    """
    child = _launch(
        """
        import os, sys, time
        from pathlib import Path

        import torch
        import torch.distributed

        from fme.core.distributed import (
            Distributed,
            add_post_shutdown_callback,
            cooperative_stop,
        )

        OUT = Path(sys.argv[1])
        rank = int(os.environ["RANK"])
        (OUT / f"pid.{rank}").write_text(str(os.getpid()))
        leave_at = 2
        add_post_shutdown_callback(
            lambda: (OUT / f"callback.{rank}").write_text("ran")
        )
        with Distributed.context():
            with cooperative_stop(budget=1.0) as stop:
                for index in range(1000):
                    torch.distributed.all_reduce(torch.ones(1))
                    time.sleep(0.02)
                    if index == leave_at:
                        if rank == 1:
                            # never reaches the boundary, and keeps its socket open
                            # until rank 0 is gone, so rank 0's death is rank 0's
                            give_up_at = time.monotonic() + 20.0
                            while not (OUT / "dead").exists():
                                if time.monotonic() > give_up_at:
                                    break
                                time.sleep(0.02)
                            os._exit(0)
                        (OUT / "raised").write_text(str(time.monotonic()))
                        raise ValueError("a NaN in the loss")
                    if stop.agreed(index):
                        break
        """,
        tmp_path,
    )
    try:
        _wait_for_file(tmp_path / "pid.0", child)
        _wait_for_file(tmp_path / "raised", child)
        died = _wait_for_death(int((tmp_path / "pid.0").read_text()), tmp_path)
    except BaseException:
        (tmp_path / "dead").write_text("go")
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        raise
    stderr = _finish(child, tmp_path)

    raised = float((tmp_path / "raised").read_text())
    # the floored exit exchange plus the teardown and the callbacks, with slack, and
    # measured against a peer that was still alive
    assert died - raised < _MIN_DEADLINE + 4.0, died - raised
    # 1, not `128 + SIGTERM`: no signal was sent, and an exception is a failure
    assert _marker_fields("hard-exit", stderr, "code") == ["1"], stderr
    # the only reason the branch used to decline to exit at all
    assert "Traceback (most recent call last)" in stderr, stderr
    assert "ValueError: a NaN in the loss" in stderr, stderr
    # the index rank 0 would have reached, which is the one its peers contribute
    assert _marker_fields("agreement-abandoned", stderr, "batch") == ["2"], stderr
    assert len(_marker_fields("shutdown-returned", stderr, "elapsed")) == 1, stderr
    assert (tmp_path / "callback.0").exists(), "the restart checkpoint was lost"


@pytest.mark.slow
@pytest.mark.medium_duration
@pytest.mark.serial
def test_cooperative_stop_is_adoptable_by_a_bare_batch_loop(tmp_path):
    """Three public names and no `Trainer`, or the seam has failed.

    The static assertion on the driver's own source is what makes this an
    adoptability test rather than another end-to-end one: it fails if the
    mechanism ever grows a dependency on the trainer, which is where a
    `Trainer._should_stop()` design would show up.

    This says nothing about whether a launcher or framework outside this
    repository can adopt it.
    """
    source = f"""
        \"\"\"A bare batch loop adopting the cooperative stop.

        **The mechanism requires this process to exit after a stop.** An exchange
        that is given up on can never be reclaimed in bounded time, so a caller
        that means to carry on in the same process is outside what the design
        bounds. This driver exits, and the assertion on its exit code is what
        holds it to that.
        \"\"\"
        import os, sys, time
        from pathlib import Path

        import torch
        import torch.distributed

        from fme.core.distributed import (
            Distributed,
            add_post_shutdown_callback,
            cooperative_stop,
        )

        OUT = Path(sys.argv[1])
        {_RECORD_EXIT}
        rank = os.environ["RANK"]
        add_post_shutdown_callback(
            lambda: (OUT / f"callback.{{rank}}").write_text("ran")
        )
        try:
            with Distributed.context():
                with cooperative_stop() as stop:
                    for i in range(1000):
                        time.sleep(0.05)
                        torch.distributed.all_reduce(torch.ones(1))
                        if i == 2:
                            (OUT / f"ready.{{rank}}").write_text("go")
                        if stop.agreed(i):
                            break
        except SystemExit as err:
            record_exit(err.code)
            raise
        """
    child = _launch(source, tmp_path)
    try:
        _wait_for_file(tmp_path / "ready.0", child)
        _wait_for_file(tmp_path / "ready.1", child)
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
    except BaseException:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        raise
    stderr = _finish(child, tmp_path)

    written = (tmp_path / "driver.py").read_text()
    assert "fme.ace" not in written
    assert "Trainer" not in written
    assert "fme.core.generics" not in written
    # the whole `fme` surface an adopter needs
    assert written.count("from fme") == 1

    batches = _marker_fields("stop-agreed", stderr, "batch")
    assert len(batches) == 2 and len(set(batches)) == 1, stderr
    assert len(_marker_fields("shutdown-returned", stderr, "elapsed")) == 2, stderr
    assert _exit_codes(tmp_path, 2) == {
        "0": str(_SIGTERM_EXIT_CODE),
        "1": str(_SIGTERM_EXIT_CODE),
    }
    assert (tmp_path / "callback.0").exists() and (tmp_path / "callback.1").exists()


@pytest.mark.slow
@pytest.mark.medium_duration
@pytest.mark.serial
@pytest.mark.parametrize(
    "backend_env",
    [
        pytest.param({"FME_DISTRIBUTED_BACKEND": "torch"}, id="torch"),
        pytest.param(
            {
                "FME_DISTRIBUTED_BACKEND": "model",
                "FME_DISTRIBUTED_H": "2",
                "FME_DISTRIBUTED_W": "1",
            },
            id="model",
        ),
    ],
)
def test_the_agreement_group_outlives_the_backend_teardown(tmp_path, backend_env):
    """The teardown must not reclaim the group, and a second world gets a new one.

    ``destroy_process_group()`` clears torch's own bookkeeping and drains nothing,
    and gloo's ``shutdown()`` is an empty default -- so whether the call returns
    turns entirely on whether it dropped the last reference to the group. Two
    independent references stop it doing so, and either alone is sufficient: the
    module global in `stop_agreement` and the backend instance's own attribute.

    Then a genuinely second ``init_process_group`` in the same process, which is the
    state `new_stop_agreement`'s cache is about and the one thing the in-session
    tests cannot build: they compare identities against ``cast``-faked worlds, so
    nothing there exercises the predicate against real torch bookkeeping. The cached
    group belongs to the destroyed world and its gloo context is no part of the new
    world's, so it must be superseded -- and moved aside rather than released, since
    releasing it runs the unbounded destructor.

    This runs in a subprocess rather than in-session, which is a departure from
    the plan: tearing the session's backend down would leave every later test in
    that rank's session without a process group, and nothing orders this test
    last. The ``model`` parametrization is what covers
    ``DistributedManager.cleanup()``.
    """
    child = _launch(
        """
        import importlib, os, sys, weakref
        from pathlib import Path

        import torch.distributed

        from fme.core.distributed import Distributed

        module = importlib.import_module("fme.core.distributed.stop_agreement")

        OUT = Path(sys.argv[1])
        with Distributed.context():
            dist = Distributed.get_instance()
            agreement = dist.stop_agreement()
            alive = weakref.ref(agreement.group)
            del agreement
            dist.shutdown()
            assert not torch.distributed.is_initialized(), "the backend is still up"
            assert alive() is not None, "the agreement group was reclaimed"

            first = module._agreement
            world_size = int(os.environ["WORLD_SIZE"])
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="file://" + str(OUT / "second-world"),
                rank=int(os.environ["RANK"]),
                world_size=world_size,
            )
            world = torch.distributed.distributed_c10d._get_default_group()
            second = module.new_stop_agreement(world_size, world)
            assert second is not first, "a destroyed world's group was handed back"
            assert module._leaked == [first], "the superseded group was not kept"
            assert second.world_size == world_size
            # and the same world asks for the same group rather than a second
            # `new_group` some ranks would not match
            assert module.new_stop_agreement(world_size, world) is second

            OUT.joinpath("survived." + str(dist.rank)).write_text("yes")
        """,
        tmp_path,
        env=_driver_env(**backend_env),
    )
    stderr = _finish(child, tmp_path)

    assert child.returncode == 0, stderr
    assert (tmp_path / "survived.0").exists() and (tmp_path / "survived.1").exists()
