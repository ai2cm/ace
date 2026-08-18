"""The termination listener must act only in the process that installed it.

Kept apart from `test_shutdown.py`, which exercises the listener with plain
forks: everything here needs a real DataLoader, real worker processes and a
real signal to the process group, so it carries a driver script and process
plumbing that the unit tests there do not.
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

    import os
    import signal
    import sys
    import time

    import torch
    import torch.utils.data

    from fme.core.distributed import Distributed

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
            record(
                "started",
                f"sigint={signal.getsignal(signal.SIGINT)!r}"
                f" start_method={torch.multiprocessing.get_start_method()}"
                f" ppid={os.getppid()}",
            )
            return torch.zeros(1)


    dist = Distributed.get_instance()
    # Production passes `instance.abort` to `abort_and_exit_on_termination`;
    # wrapping it here observes which process ran it, without touching
    # production code.
    backend_abort = dist.abort
    dist.abort = lambda: (record("abort"), backend_abort())
    # the context only installs the listener for real multi-rank jobs, and this
    # driver is a single process standing in for one rank of one
    dist.is_distributed = lambda: True

    # the real entrypoint context, so the listener is installed by production
    # code rather than by the test
    with Distributed.context():
        loader = torch.utils.data.DataLoader(
            _Dataset(),
            batch_size=1,
            num_workers=2,
            # Mirrors the arguments fme/ace/data_loading/getters.py gives the
            # loader on the non-zarr path: the default start method (fork, on
            # Linux), no worker initializer, non-persistent workers.
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

# The driver is a separate interpreter of the same build, so its default matches.
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
        "worker inherits the listener setup this test checks is disarmed"
    ),
)
def test_dataloader_workers_do_not_abort_when_the_group_is_signalled(tmp_path):
    """Only the process that installed the listener may abort and exit.

    The default DataLoader start method is fork, so every worker inherits the
    ignore-handler and the wakeup fd, and torchrun signals the whole process
    group, delivering the signal to the workers directly. Without the at-fork
    disarm a worker's copy of the wakeup fd feeds the parent's pipe, and the
    parent cannot tell a signal sent to a worker from its own preemption.

    SIGINT is the signal to test with: torch replaces the SIGTERM disposition
    in every worker with a C-level handler of its own
    (`signal_handling._set_worker_signal_handlers`, called at the top of
    `_worker_loop`, after the at-fork hooks), so a SIGTERM case could pass
    whether or not the disarm ran. It installs nothing for SIGINT. Ctrl-C
    delivers SIGINT to the whole foreground group, and torchrun's agent
    forwards whichever death signal it received.
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

        child.wait(timeout=_EXIT_TIMEOUT)
    finally:
        _kill_group(pgid)
        child.wait(timeout=_EXIT_TIMEOUT)
        if child.stderr is not None:
            child.stderr.close()

    # Preconditions, so that the assertions below cannot pass vacuously: two
    # workers really were running, in their own processes, fork-started inside
    # the armed context.
    worker_pids = set(started) - {driver_pid}
    assert len(worker_pids) >= 2, f"expected forked workers, got {started}"
    for pid in worker_pids:
        assert (
            "start_method=fork" in started[pid]
        ), f"worker {pid} was not fork-started: {started[pid]}"
        # the at-fork disarm restored the default disposition; the parent's
        # routing handler would read as `_route_to_listener` here
        assert (
            "sigint=<Handlers.SIG_DFL" in started[pid]
        ), f"worker {pid} was not disarmed at fork: {started[pid]}"

    aborts = _markers(str(marker_dir), "abort")
    assert (
        driver_pid in aborts
    ), f"the driver itself did not abort on the signal: {aborts=}"
    assert set(aborts) == {
        driver_pid
    }, f"a process other than the driver ran the abort: {aborts}"
    assert child.returncode == 128 + sig
