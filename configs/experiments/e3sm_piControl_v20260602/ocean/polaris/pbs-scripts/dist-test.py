"""Minimal 2-node distributed connectivity test.

Mirrors the training launch (torchrun, NCCL, static rendezvous) but does only
init_process_group + one all_reduce, with a short timeout so it fails fast
instead of hanging. Use this to isolate whether multi-node rendezvous/NCCL is
the problem, independent of fme/checkpoint/data loading.
"""

import datetime
import os
import socket

import torch
import torch.distributed as dist


def main():
    host = socket.gethostname()
    info = {k: os.environ.get(k) for k in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK",
        "MASTER_ADDR", "MASTER_PORT",
    )}
    print(f"[{host}] start pid={os.getpid()} env={info}", flush=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[{host}] cuda.set_device({local_rank}) ok, "
          f"device={torch.cuda.get_device_name(local_rank)}", flush=True)

    # Short timeout: if rendezvous/NCCL can't connect, error out in ~2 min.
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=120)
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(f"[{host}] init_process_group OK rank={rank}/{world} "
          f"local_rank={local_rank}", flush=True)

    t = torch.ones(1, device=f"cuda:{local_rank}") * rank
    dist.all_reduce(t)
    expected = sum(range(world))
    print(f"[{host}] all_reduce result={t.item():.0f} expected={expected} "
          f"rank={rank}", flush=True)

    dist.barrier()
    if rank == 0:
        ok = int(t.item()) == expected
        print(f"DIST TEST {'PASSED' if ok else 'FAILED(value)'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
