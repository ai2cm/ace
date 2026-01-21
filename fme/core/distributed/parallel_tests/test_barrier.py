"""
Test that barrier synchronization works under different modes of parallelism.
"""

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE, requires_parallel


def test_barrier_completes(monkeypatch):
    """All ranks should pass through barrier() without deadlock.

    For NonDistributed, the barrier is a no-op
    """
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(WORLD_SIZE))

    dist = Distributed.get_instance()
    device = get_device()

    # Each rank writes to its own tensor, then barriers, then reads.
    # If barrier works, all ranks survive; if not, torchrun will timeout.
    local_tensor = torch.full(
        (1,), fill_value=float(dist.rank), dtype=torch.float32, device=device
    )
    dist.barrier()
    # After barrier, verify our local tensor is untouched
    assert local_tensor.item() == float(dist.rank)


@requires_parallel
@pytest.mark.parametrize(
    "is_h_parallel,is_w_parallel, use_world_size",
    [
        (1, 0, 1),  # H-parallel split
        (0, 1, 0),  # W-parallel split
        (0, 0, 0),  # automatic data-parallel split
        (1, 1, 0),  # both
    ],
)
def test_barrier_all_gather_visibility(
    is_h_parallel, is_w_parallel, use_world_size, monkeypatch
):
    """After barrier, all ranks can see each other's tensors.

    Each rank creates a small tensor filled with its rank value, calls
    `dist.barrier()`, then performs an `all_gather` across the world group and
    verifies that the gathered tensor contains all ranks' values in order.

    This asserts both barrier synchronization and cross-rank communication.
    """

    # manipulate the size to expand coverage
    use_size = WORLD_SIZE if use_world_size else 2
    h_size = use_size if is_h_parallel else 1
    w_size = use_size if is_w_parallel else 1

    if WORLD_SIZE % (h_size * w_size) != 0:
        pytest.skip(
            f"world_size={WORLD_SIZE} not divisible by " f"(h={h_size} * w={w_size})"
        )

    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_size))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_size))

    dist = Distributed.get_instance()
    device = get_device()

    # local tensor with two elements so gather shows repetition per rank
    local = torch.full(
        (2,), fill_value=float(dist.rank), dtype=torch.float32, device=device
    )

    # Ensure all ranks reach the barrier
    dist.barrier()

    world_size = torch.distributed.get_world_size()
    world_group = dist.comm_get_group("world")
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local, group=world_group)
    gathered_tensor = torch.cat(gathered, dim=0)

    expected = torch.cat(
        [
            torch.full((2,), float(r), dtype=torch.float32, device=device)
            for r in range(world_size)
        ],
        dim=0,
    )

    torch.testing.assert_close(gathered_tensor, expected)
