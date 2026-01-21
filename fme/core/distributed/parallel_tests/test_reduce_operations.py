"""
Tests for reduce_sum, reduce_min, and reduce_max under spatial parallelism.

These tests run serially (pytest) and in parallel (torchrun), on CPU and GPU.
"""

import os

import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import requires_parallel


@requires_parallel
def test_reduce_sum(monkeypatch):
    """reduce_sum should sum tensors across all ranks."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    # Each rank contributes a tensor of ones
    local_tensor = torch.ones((2, 2), dtype=torch.float32, device=device)

    result = dist.reduce_sum(local_tensor)

    expected = torch.full_like(result, float(world_size))
    torch.testing.assert_close(result, expected)


@requires_parallel
def test_reduce_min(monkeypatch):
    """reduce_min should return the element-wise minimum across all ranks."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    # Each rank contributes its rank index as fill value
    local_tensor = torch.full(
        (2, 2), fill_value=float(dist.rank), dtype=torch.float32, device=device
    )

    result = dist.reduce_min(local_tensor)

    # Minimum across all ranks should be 0 (rank 0's value)
    expected = torch.zeros_like(result)
    torch.testing.assert_close(result, expected)


@requires_parallel
def test_reduce_max(monkeypatch):
    """reduce_max should return the element-wise maximum across all ranks."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    # Each rank contributes its rank index as fill value
    local_tensor = torch.full(
        (2, 2), fill_value=float(dist.rank), dtype=torch.float32, device=device
    )

    result = dist.reduce_max(local_tensor)

    # Maximum across all ranks should be world_size - 1 (last rank's value)
    expected = torch.full_like(result, float(world_size - 1))
    torch.testing.assert_close(result, expected)
