"""Empirical probe: on SIGTERM, does aborting the NCCL communicators from a
non-main thread quiesce this rank's fabric traffic, and leave the GPU fabric
healthy after every rank has exited?

This is the load-bearing question for replacing the collective teardown of
ai2cm/ace#1398 with communicator abort. Both scenarios run a real
DistributedDataParallel training loop, so the collectives in flight are DDP's
bucketed gradient all-reduces, issued from autograd hooks on side streams as
in production. Each is followed by a fresh NCCL job on the same GPUs as a
fabric health check:

- ``healthy``: every rank is mid training loop when SIGTERM arrives — the
  common preemption, where the abort races active collectives.
- ``wedge``: rank 0 stops training while its peers run ``backward()``, whose
  gradient all-reduces it never joins, so the peers' main threads are blocked
  in a stream sync and cannot service a signal — the pathological preemption
  that cordons nodes today.

Run directly on a multi-GPU machine::

    python fme/core/distributed/nccl_abort_probe.py --nproc 8

The ``--no-abort`` arm reproduces today's failure (a rank exits while a peer's
kernel still polls its memory over NVLink) and is expected to fault the fabric
on NVSwitch nodes. Do not run it on a shared node.
"""

import argparse
import os
import platform
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
# 32 MiB, comparable to a DDP gradient bucket: large enough for the
# Simple/NVLS protocols a training job uses, not the small-message LL path.
_TENSOR_NUMEL = 1 << 23
# ~17M parameters -> ~67 MB of gradients, several DDP buckets at the default
# 25 MB bucket cap.
_MODEL_WIDTH = 2048
_MODEL_DEPTH = 4
_BATCH_SIZE = 8


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
    tensor = torch.ones(_TENSOR_NUMEL, device=torch.device("cuda", local_rank))
    return rank, tensor


def _training_setup() -> (
    tuple[int, torch.nn.Module, torch.optim.Optimizer, torch.Tensor]
):
    rank, _ = _init_nccl()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    layers: list[torch.nn.Module] = []
    for _ in range(_MODEL_DEPTH):
        layers.extend([torch.nn.Linear(_MODEL_WIDTH, _MODEL_WIDTH), torch.nn.GELU()])
    model = torch.nn.parallel.DistributedDataParallel(
        torch.nn.Sequential(*layers).to(device), device_ids=[device.index]
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    batch = torch.randn(_BATCH_SIZE, _MODEL_WIDTH, device=device)
    return rank, model, optimizer, batch


def _training_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor
) -> None:
    optimizer.zero_grad()
    loss = model(batch).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


def _run_healthy_worker(ready_dir: Path, grace: float, abort_comms: bool) -> None:
    rank = int(os.environ["RANK"])
    _install_sigterm_listener(rank, grace, abort_comms)
    rank, model, optimizer, batch = _training_setup()
    iterations = 0
    while True:
        try:
            _training_step(model, optimizer, batch)
        except BaseException as exc:
            _mark(f"rank {rank} loop raised {type(exc).__name__} after abort")
            time.sleep(_NEVER)  # the listener owns the exit
        iterations += 1
        if iterations == 3:
            _mark(f"rank {rank} loop running")
            (ready_dir / f"looping_rank{rank}").touch()


def _run_wedge_worker(ready_dir: Path, grace: float, abort_comms: bool) -> None:
    rank = int(os.environ["RANK"])
    if not abort_comms:
        # Make the fault deterministic: rank 0 exits while the wedged ranks'
        # kernels are still polling its memory.
        grace = 1.0 if rank == 0 else 10.0
    _install_sigterm_listener(rank, grace, abort_comms)
    rank, model, optimizer, batch = _training_setup()
    for _ in range(3):
        _training_step(model, optimizer, batch)
    _mark(f"rank {rank} warmup complete")
    (ready_dir / f"warmup_rank{rank}").touch()
    if rank == 0:
        # Stand in for the rank that left the training loop first: its peers
        # run a backward whose gradient all-reduces this rank never joins.
        time.sleep(_NEVER)
    else:
        (ready_dir / f"wedging_rank{rank}").touch()
        _mark(f"rank {rank} entering wedge backward")
        try:
            _training_step(model, optimizer, batch)
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


def _print_node_info(moment: str) -> None:
    node_id = os.environ.get("BEAKER_NODE_ID", "")
    sys.stdout.write(f"\n----- node ({moment}) -----\n")
    sys.stdout.write(f"hostname: {platform.node()}  BEAKER_NODE_ID: {node_id}\n")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        # topology tells us whether a pass is fabric evidence (P2P/NVLink) or
        # only mechanism evidence; link error counters catch a fault that a
        # succeeding healthcheck job would mask
        for args in (["topo", "-m"], ["nvlink", "-e"]):
            result = subprocess.run(
                [nvidia_smi, *args], capture_output=True, text=True, check=False
            )
            sys.stdout.write(f"\n$ nvidia-smi {' '.join(args)}\n{result.stdout}\n")
    sys.stdout.flush()


def _expected_ready(scenario: str, nproc: int) -> set[str]:
    if scenario == "healthy":
        return {f"looping_rank{rank}" for rank in range(nproc)}
    ready = {f"warmup_rank{rank}" for rank in range(nproc)}
    return ready | {f"wedging_rank{rank}" for rank in range(1, nproc)}


def _run_scenario_job(
    scenario: str, nproc: int, grace: float, abort_comms: bool, work_dir: Path
) -> list[str]:
    """Run one preemption scenario, SIGTERM it, and return failures."""
    failures: list[str] = []
    ready_dir = work_dir / "ready"
    ready_dir.mkdir()
    log_path = work_dir / f"{scenario}.log"
    script_args = [
        f"--role={scenario}",
        f"--ready-dir={ready_dir}",
        f"--grace={grace}",
    ]
    if not abort_comms:
        script_args.append("--no-abort")
    env = os.environ | {"NCCL_DEBUG": "INFO"}  # shows which transport was used
    expected_ready = _expected_ready(scenario, nproc)
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
                failures.append(f"{scenario} job exited before it was signalled")
            elif not expected_ready <= {path.name for path in ready_dir.iterdir()}:
                failures.append(f"{scenario} job did not reach readiness in 120s")
            else:
                time.sleep(2.0)  # let the ranks settle into the scenario
                # SIGTERM to torchrun, which forwards it to every rank -- the
                # same path a scheduler preemption takes.
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=90.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    log_text = _print_log(f"{scenario} job log", log_path)
    if failures:
        return failures
    for rank in range(nproc):
        for event in ["listener received SIGTERM", "exiting 143"] + (
            ["abort returned"] if abort_comms else []
        ):
            if f"rank {rank} {event}" not in log_text:
                failures.append(f"{scenario}: rank {rank}: no '{event}' marker")
    if scenario == "wedge":
        for rank in range(1, nproc):
            if f"rank {rank} main thread released from wedge" not in log_text:
                failures.append(
                    f"wedge: rank {rank}: main thread never released from wedge"
                )
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
                f"{scenario}: a rank exited before every abort had returned; "
                "the grace period is too short"
            )
    return failures


def _run_healthcheck_job(scenario: str, nproc: int, work_dir: Path) -> list[str]:
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
            returncode: int | None = proc.wait(timeout=180.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = None
    log_text = _print_log(f"healthcheck log (after {scenario})", log_path)
    if returncode != 0:
        failures.append(
            f"healthcheck after {scenario} failed with returncode {returncode}"
        )
    for rank in range(nproc):
        if f"rank {rank} healthcheck ok" not in log_text:
            failures.append(f"healthcheck after {scenario}: no rank {rank} marker")
    return failures


def run_probe(
    nproc: int,
    work_dir: Path,
    grace: float = 3.0,
    abort_comms: bool = True,
    scenarios: tuple[str, ...] = ("healthy", "wedge"),
) -> bool:
    """Run each scenario plus its fabric health check; return whether all passed."""
    _print_node_info("start")
    failures: list[str] = []
    for scenario in scenarios:
        scenario_dir = work_dir / scenario
        scenario_dir.mkdir()
        failures += _run_scenario_job(scenario, nproc, grace, abort_comms, scenario_dir)
        failures += _run_healthcheck_job(scenario, nproc, scenario_dir)
    _print_node_info("end")
    if failures:
        for failure in failures:
            sys.stdout.write(f"{_PREFIX} FAILURE: {failure}\n")
        sys.stdout.write(f"{_PREFIX} VERDICT: FAIL\n")
    else:
        sys.stdout.write(
            f"{_PREFIX} VERDICT: PASS -- every rank saw SIGTERM on the listener "
            "thread and aborted before any rank exited, and a fresh NCCL job on "
            f"the same GPUs succeeded after each of: {', '.join(scenarios)}\n"
        )
    sys.stdout.flush()
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=["driver", "healthy", "wedge", "healthcheck"],
        default="driver",
    )
    parser.add_argument("--nproc", type=int, default=torch.cuda.device_count())
    parser.add_argument("--grace", type=float, default=3.0)
    parser.add_argument(
        "--no-abort",
        action="store_true",
        help="control arm reproducing the fabric fault; do not run on a " "shared node",
    )
    parser.add_argument("--ready-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.role == "healthy":
        _run_healthy_worker(args.ready_dir, args.grace, not args.no_abort)
        return 0
    if args.role == "wedge":
        _run_wedge_worker(args.ready_dir, args.grace, not args.no_abort)
        return 0
    if args.role == "healthcheck":
        _run_healthcheck_worker()
        return 0
    if args.nproc < 2:
        raise SystemExit("the probe needs at least two GPUs")
    scenarios = ("wedge",) if args.no_abort else ("healthy", "wedge")
    with tempfile.TemporaryDirectory() as work_dir:
        passed = run_probe(
            args.nproc, Path(work_dir), args.grace, not args.no_abort, scenarios
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
