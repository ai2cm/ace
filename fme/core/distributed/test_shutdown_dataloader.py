"""The termination handler must act only in the process that installed it.

Kept apart from `test_shutdown.py`, which exercises the handler in-process:
everything here needs a real DataLoader, real worker processes and a real
process group to signal, so it carries a driver script and process plumbing
that the unit tests there do not.
"""

import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

# torchrun's elastic agent puts each rank in its own session and signals the
# whole group (`SubprocessHandler`: `start_new_session=True`, then
# `os.killpg(self.proc.pid, death_sig)`), so the driver is launched the same
# way and signalled the same way.
_DRIVER = textwrap.dedent(
    '''
    """Run a DataLoader with forked workers inside a real Distributed.context().

    Records what each process did as marker files named `<event>.<pid>`, the
    only channel a forked DataLoader worker has back to the test.
    """

    import multiprocessing
    import os
    import signal
    import sys
    import time

    import torch
    import torch.utils.data

    from fme.core.distributed import Distributed
    from fme.core.distributed.shutdown import add_post_shutdown_callback

    MARKER_DIR = sys.argv[1]
    MAIN_PID = os.getpid()
    WORKERS_STARTED_TIMEOUT = 60.0


    def record(event, detail=""):
        # Readers poll by filename and `open(path, "w")` publishes the name
        # before the detail, so rename in. The temp name must not start with
        # the event, or they match it.
        path = os.path.join(MARKER_DIR, f"{event}.{os.getpid()}")
        partial = os.path.join(MARKER_DIR, f".partial.{event}.{os.getpid()}")
        with open(partial, "w") as f:
            f.write(detail)
            f.flush()
            os.fsync(f.fileno())
        os.replace(partial, path)


    def started_worker_pids():
        return {
            name.rsplit(".", 1)[-1]
            for name in os.listdir(MARKER_DIR)
            if name.startswith("started.")
        } - {str(MAIN_PID)}


    class _Dataset(torch.utils.data.Dataset):
        """Reports, from whichever process fetches a sample, what it inherited."""

        def __len__(self):
            return 1 << 20

        def __getitem__(self, index):
            dispositions = " ".join(
                f"{name}={getattr(signal.getsignal(sig), '__qualname__', None)!r}"
                for name, sig in (
                    ("sigint", signal.SIGINT),
                    ("sigterm", signal.SIGTERM),
                )
            )
            record(
                "started",
                f"{dispositions}"
                f" start_method={multiprocessing.get_start_method()}"
                f" ppid={os.getppid()}",
            )
            return torch.zeros(1)


    dist = Distributed.get_instance()
    # Production passes `instance.shutdown` to `handle_termination_signals`;
    # wrapping it here observes the backend teardown itself, not just the
    # callbacks that follow it, and without touching production code.
    backend_shutdown = dist.shutdown
    dist.shutdown = lambda: (record("shutdown"), backend_shutdown())

    # the real entrypoint context, so the handler is installed by production
    # code rather than by the test
    with Distributed.context():
        # stands in for the Trainer's restart-checkpoint write
        add_post_shutdown_callback(lambda: record("callback"))

        loader = torch.utils.data.DataLoader(
            _Dataset(),
            batch_size=1,
            num_workers=2,
            # Mirrors the arguments fme/ace/data_loading/getters.py:116-117
            # gives the loader on the non-zarr path: the default start method
            # (fork, on Linux), no worker initializer to reset the inherited
            # signal dispositions, and non-persistent workers.
            multiprocessing_context=None,
            worker_init_fn=None,
            persistent_workers=False,
        )
        batches = iter(loader)
        give_up_at = time.monotonic() + WORKERS_STARTED_TIMEOUT
        while len(started_worker_pids()) < 2:
            if time.monotonic() > give_up_at:
                raise AssertionError("the DataLoader workers never started")
            next(batches)

        record("ready", " ".join(sorted(started_worker_pids())))
        # hold the workers alive and idle, as a rank does while it trains
        while True:
            time.sleep(0.05)
    '''
)

# torchrun's own environment would send the driver down the TorchDistributed
# path and block it on a rendezvous that never happens.
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

# Generous, because they only decide which diagnostic a hang produces, not how
# long one takes to notice: the suite's autouse 90s alarm (see conftest.py) fires
# first either way. Lowering them would trade its "failed due to timeout" for our
# own message without making either arrive sooner. Observed runtime is ~10s.
_READY_TIMEOUT = 120.0
_EXIT_TIMEOUT = 120.0
# A worker that decides to tear down writes its marker in microseconds, so this
# is long enough to catch one. It is not a guarantee: the driver's exit joins
# each worker for only MP_STATUS_CHECK_INTERVAL (5s) before terminating it, so
# the driver being gone does not mean a worker finished arbitrary work -- the
# negative assertion below rests on this sleep, not on that join.
_SETTLE = 2.0

_HANDLER_QUALNAME = "handle_termination_signals.<locals>.handle"

# The behavior under test only exists where a worker starts by forking, so that
# it inherits an already-installed handler. Asking for fork explicitly on a
# platform that does not default to it would test a loader production never
# builds, so the precondition is read rather than forced: fork on Linux under
# the Python versions this runs on, spawn on macOS, and forkserver on Linux from
# Python 3.14. A spawned or forkserver-started worker begins from a fresh
# interpreter with no handler to inherit, leaving nothing to decline to act on.
# It also cannot run this driver, which arrives as `python -c` and so has a
# `__main__` that a re-importing worker cannot load `_Dataset` from.
_DEFAULT_START_METHOD = multiprocessing.get_start_method()


def _markers(marker_dir: str, event: str) -> dict[str, str]:
    """Map pid to detail for every marker file recording `event`."""
    found = {}
    for name in os.listdir(marker_dir):
        recorded_event, _, pid = name.rpartition(".")
        if recorded_event == event:
            with open(os.path.join(marker_dir, name)) as f:
                found[pid] = f.read()
    return found


def _wait_for(
    marker_dir: str, event: str, child: "subprocess.Popen[str]", timeout: float
) -> dict[str, str]:
    give_up_at = time.monotonic() + timeout
    while True:
        found = _markers(marker_dir, event)
        if found:
            return found
        if child.poll() is not None:
            stderr = child.stderr.read() if child.stderr is not None else ""
            raise AssertionError(
                f"the driver exited with {child.returncode} before recording "
                f"{event!r}; stderr:\n{stderr}"
            )
        if time.monotonic() > give_up_at:
            raise AssertionError(
                f"timed out waiting for the driver to record {event!r}"
            )
        time.sleep(0.05)


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.medium_duration
@pytest.mark.skipif(
    _DEFAULT_START_METHOD != "fork",
    reason=(
        f"DataLoader workers default to {_DEFAULT_START_METHOD} here, so no "
        "worker inherits the handler this test checks is declined"
    ),
)
def test_dataloader_workers_do_not_tear_down_when_the_group_is_signalled(tmp_path):
    """Only the process that installed the handler may run the teardown.

    Where the default DataLoader start method is fork, every worker inherits
    the handler, and torchrun signals the whole process group, which delivers
    the signal to the workers directly. A worker that runs the teardown destroys a
    fork-inherited process group and re-runs the parent's callbacks, including
    the Trainer's multi-GB restart-checkpoint write, which then races the
    parent's own write of the same path.

    The `get_worker_info` guard in `handle_termination_signals` cannot prevent
    this: it is consulted when the context is entered, not when the signal
    arrives, and a forked worker inherited an already-installed handler.

    SIGINT is the signal to test with, because it is the one that reaches the
    workers' Python handler. torch replaces the SIGTERM disposition in every
    worker with a C-level handler of its own
    (`signal_handling._set_worker_signal_handlers`, called at the top of
    `_worker_loop`), which pre-empts the Python-level handler and lets the
    worker die instead; it installs nothing for SIGINT. A SIGTERM case would
    therefore pass whether or not the handler guards itself, so it is left
    out. Ctrl-C delivers SIGINT to the whole foreground group, and torchrun's
    agent forwards whichever death signal it received.
    """
    sig = signal.SIGINT
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in _RANK_ENV}
    env.pop("FME_DISTRIBUTED_BACKEND", None)
    env["FME_FORCE_CPU"] = "1"

    child = subprocess.Popen(
        [sys.executable, "-c", _DRIVER, str(marker_dir)],
        start_new_session=True,  # as torchrun launches a rank
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    pgid = os.getpgid(child.pid)
    driver_pid = str(child.pid)
    try:
        _wait_for(str(marker_dir), "ready", child, _READY_TIMEOUT)
        started = _markers(str(marker_dir), "started")

        os.killpg(pgid, sig)  # as torchrun kills a rank

        _wait_for(str(marker_dir), "callback", child, _EXIT_TIMEOUT)
        try:
            child.wait(timeout=_EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(_SETTLE)
    finally:
        _kill_group(pgid)
        child.wait(timeout=_EXIT_TIMEOUT)
        if child.stderr is not None:
            child.stderr.close()

    # Preconditions, so that the assertions below cannot pass vacuously: two
    # workers really were running, in their own processes, and really did
    # inherit the handler by forking.
    worker_pids = set(started) - {driver_pid}
    assert len(worker_pids) >= 2, f"expected forked workers, got {started}"
    inherited = f"{sig.name.lower()}='{_HANDLER_QUALNAME}'"
    for pid in worker_pids:
        assert (
            inherited in started[pid]
        ), f"worker {pid} did not inherit the {sig.name} handler: {started[pid]}"
        assert (
            "start_method=fork" in started[pid]
        ), f"worker {pid} was not fork-started: {started[pid]}"

    shutdowns = _markers(str(marker_dir), "shutdown")
    callbacks = _markers(str(marker_dir), "callback")
    # and the signal really did reach the group, so an absent worker teardown
    # means the worker declined to act, not that nothing was ever signalled
    assert (
        driver_pid in shutdowns and driver_pid in callbacks
    ), f"the driver itself did not tear down: {shutdowns=} {callbacks=}"

    assert set(shutdowns) == {
        driver_pid
    }, f"a process other than the driver shut the backend down: {shutdowns}"
    assert set(callbacks) == {driver_pid}, (
        f"a process other than the driver ran the post-shutdown callbacks: "
        f"{callbacks}"
    )
