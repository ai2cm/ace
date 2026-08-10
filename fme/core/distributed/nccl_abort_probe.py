"""Empirical probe: on SIGTERM, does aborting the NCCL communicators from a
non-main thread release a rank wedged inside a collective, and leave the GPU
fabric healthy after every rank has exited?

This is the load-bearing question for replacing the collective teardown of
ai2cm/ace#1398 with communicator abort. The scenario reproduces the preemption
state that cordons nodes: rank 0 abandons a collective its peers have entered,
so the peers' main threads are blocked in a stream sync and cannot service a
signal; SIGTERM then arrives at every rank, as it does when the scheduler
preempts the job.

Run directly on a multi-GPU machine::

    python fme/core/distributed/nccl_abort_probe.py --nproc 8

The ``--no-abort`` arm reproduces today's failure (a rank exits while a peer's
kernel still polls its memory over NVLink) and is expected to fault the fabric
on NVSwitch nodes. Do not run it on a shared node.
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

_PREFIX = "abort-probe:"
_NEVER = 3600.0


def _mark(message: str) -> None:
    # os.write, not print/logging: it must work from a rank whose main thread
    # is wedged and may hold arbitrary locks.
    os.write(2, f"{_PREFIX} {message}\n".encode())


def _install_sigterm_listener(rank: int, grace: float, abort_comms: bool) -> None:
    """Respond to SIGTERM on a dedicated thread, so the response does not
    depend on the main thread ever returning to the interpreter.

    Not pthread_sigmask + sigwait: threads that already exist when the mask is
    set (torch spawns many at import) keep SIGTERM unblocked, and a
    process-directed signal delivered to any of them takes the default action.
    The wakeup fd hears the signal no matter which thread receives it.
    """
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
    # the no-op handler makes CPython's C-level handler catch SIGTERM (keeping
    # the process alive) and write to the wakeup fd; the response itself
    # belongs to the listener thread
    signal.signal(signal.SIGTERM, lambda signum, frame: None)

    def listen() -> None:
        while signal.SIGTERM not in os.read(read_fd, 64):
            pass
        _mark(f"rank {rank} listener received SIGTERM")
        if abort_comms:
            start = time.monotonic()
            dist.distributed_c10d._abort_process_group()
            elapsed = time.monotonic() - start
            _mark(f"rank {rank} abort returned after {elapsed:.2f}s")
        # Exiting revokes this GPU's peer-memory mappings, so give the peers'
        # own aborts time to kill any kernel still polling them.
        time.sleep(grace)
        _mark(f"rank {rank} exiting 143")
        os._exit(143)

    threading.Thread(target=listen, name="sigterm-listener", daemon=True).start()


def _init_nccl() -> tuple[int, torch.Tensor]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    tensor = torch.ones(1 << 16, device=torch.device("cuda", local_rank))
    return rank, tensor


def _run_wedge_worker(ready_dir: Path, grace: float, abort_comms: bool) -> None:
    rank = int(os.environ["RANK"])
    if not abort_comms:
        # Make the fault deterministic: rank 0 exits while the wedged ranks'
        # kernels are still polling its memory.
        grace = 1.0 if rank == 0 else 10.0
    _install_sigterm_listener(rank, grace, abort_comms)
    rank, tensor = _init_nccl()
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    _mark(f"rank {rank} warmup complete")
    (ready_dir / f"warmup_rank{rank}").touch()
    if rank == 0:
        # Stand in for the rank that left the training loop first: its peers
        # enter a collective this rank never joins.
        time.sleep(_NEVER)
    else:
        (ready_dir / f"wedging_rank{rank}").touch()
        _mark(f"rank {rank} entering wedge all-reduce")
        try:
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            _mark(f"rank {rank} main thread released from wedge")
        except BaseException as exc:
            _mark(
                f"rank {rank} main thread released from wedge "
                f"with {type(exc).__name__}: {exc}"
            )
        time.sleep(_NEVER)  # the listener owns the exit


def _run_healthcheck_worker() -> None:
    rank, tensor = _init_nccl()
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    expected = float(dist.get_world_size())
    if not torch.allclose(tensor, torch.full_like(tensor, expected)):
        raise RuntimeError(f"all-reduce returned wrong values on rank {rank}")
    _mark(f"rank {rank} healthcheck ok")
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _torchrun_command(nproc: int, script_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={nproc}",
        "--master-addr=127.0.0.1",
        f"--master-port={_free_port()}",
        str(Path(__file__).resolve()),
        *script_args,
    ]


def _print_log(title: str, log_path: Path) -> str:
    text = log_path.read_text(errors="replace")
    sys.stdout.write(f"\n----- {title} -----\n{text}\n----- end {title} -----\n")
    sys.stdout.flush()
    return text


def _print_topology() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return
    result = subprocess.run(
        [nvidia_smi, "topo", "-m"], capture_output=True, text=True, check=False
    )
    sys.stdout.write(f"\n----- GPU topology -----\n{result.stdout}\n")
    sys.stdout.flush()


def _run_wedge_job(
    nproc: int, grace: float, abort_comms: bool, work_dir: Path
) -> tuple[str, list[str]]:
    """Run the wedge scenario, SIGTERM it, and return (log text, failures)."""
    failures: list[str] = []
    ready_dir = work_dir / "ready"
    ready_dir.mkdir()
    log_path = work_dir / "wedge.log"
    script_args = ["--role=wedge", f"--ready-dir={ready_dir}", f"--grace={grace}"]
    if not abort_comms:
        script_args.append("--no-abort")
    env = os.environ | {"NCCL_DEBUG": "INFO"}  # shows which transport was used
    expected_ready = {f"warmup_rank{rank}" for rank in range(nproc)}
    expected_ready |= {f"wedging_rank{rank}" for rank in range(1, nproc)}
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            _torchrun_command(nproc, script_args),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                ready = {path.name for path in ready_dir.iterdir()}
                if expected_ready <= ready or proc.poll() is not None:
                    break
                time.sleep(0.5)
            if proc.poll() is not None:
                failures.append("wedge job exited before it was signalled")
            elif not expected_ready <= {path.name for path in ready_dir.iterdir()}:
                failures.append("wedge job did not reach the wedge in 120s")
            else:
                time.sleep(2.0)  # let the wedged ranks block in the stream sync
                # SIGTERM to torchrun, which forwards it to every rank -- the
                # same path a scheduler preemption takes.
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=90.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    log_text = _print_log("wedge job log", log_path)
    if not failures:
        for rank in range(nproc):
            for event in ["listener received SIGTERM", "exiting 143"] + (
                ["abort returned"] if abort_comms else []
            ):
                if f"rank {rank} {event}" not in log_text:
                    failures.append(f"rank {rank}: no '{event}' marker")
        for rank in range(1, nproc):
            if f"rank {rank} main thread released from wedge" not in log_text:
                failures.append(f"rank {rank}: main thread never released from wedge")
    if not failures and abort_comms:
        # The fault condition is a process dying while a peer's kernel still
        # holds mappings into its memory, so no rank may exit before every
        # rank's abort has returned. All ranks share this log via torchrun.
        last_abort = max(
            log_text.rfind(f"rank {rank} abort returned") for rank in range(nproc)
        )
        first_exit = min(
            log_text.find(f"rank {rank} exiting 143") for rank in range(nproc)
        )
        if first_exit < last_abort:
            failures.append(
                "a rank exited before every abort had returned; "
                "the grace period is too short"
            )
    return log_text, failures


def _run_healthcheck_job(nproc: int, work_dir: Path) -> list[str]:
    """Run a fresh NCCL job on the same GPUs and return failures."""
    failures: list[str] = []
    log_path = work_dir / "healthcheck.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            _torchrun_command(nproc, ["--role=healthcheck"]),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            returncode = proc.wait(timeout=180.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = None
    log_text = _print_log("healthcheck job log", log_path)
    if returncode != 0:
        failures.append(f"healthcheck job failed with returncode {returncode}")
    for rank in range(nproc):
        if f"rank {rank} healthcheck ok" not in log_text:
            failures.append(f"rank {rank}: no healthcheck marker")
    return failures


def run_probe(
    nproc: int, work_dir: Path, grace: float = 3.0, abort_comms: bool = True
) -> bool:
    """Run the full probe; print a verdict and return whether it passed."""
    _print_topology()
    _, failures = _run_wedge_job(nproc, grace, abort_comms, work_dir)
    failures += _run_healthcheck_job(nproc, work_dir)
    if failures:
        for failure in failures:
            sys.stdout.write(f"{_PREFIX} FAILURE: {failure}\n")
        sys.stdout.write(f"{_PREFIX} VERDICT: FAIL\n")
    else:
        sys.stdout.write(
            f"{_PREFIX} VERDICT: PASS -- every rank saw SIGTERM on the listener "
            "thread, wedged ranks were released, and a fresh NCCL job on the "
            "same GPUs succeeded\n"
        )
    sys.stdout.flush()
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=["driver", "wedge", "healthcheck"], default="driver"
    )
    parser.add_argument("--nproc", type=int, default=torch.cuda.device_count())
    parser.add_argument("--grace", type=float, default=3.0)
    parser.add_argument(
        "--no-abort",
        action="store_true",
        help="control arm reproducing the fabric fault; do not " "run on a shared node",
    )
    parser.add_argument("--ready-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.role == "wedge":
        _run_wedge_worker(args.ready_dir, args.grace, not args.no_abort)
        return 0
    if args.role == "healthcheck":
        _run_healthcheck_worker()
        return 0
    if args.nproc < 2:
        raise SystemExit("the probe needs at least two GPUs")
    with tempfile.TemporaryDirectory() as work_dir:
        passed = run_probe(args.nproc, Path(work_dir), args.grace, not args.no_abort)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
