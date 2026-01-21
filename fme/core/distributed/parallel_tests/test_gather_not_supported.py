"""
Negative tests: verify that gather and gather_irregular raise NotImplementedError
under spatial parallelism (ModelTorchDistributed backend).

TODO: when impl'ed, these tests will be updated to reflect that
"""

import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import requires_parallel


@requires_parallel
def test_gather_raises_not_implemented(monkeypatch):
    """ModelTorchDistributed.gather() should raise NotImplementedError."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    tensor = torch.ones((2, 2), dtype=torch.float32, device=device)

    with pytest.raises(NotImplementedError, match="gather.*spatial parallelism"):
        dist.gather(tensor)


@requires_parallel
def test_gather_irregular_raises_not_implemented(monkeypatch):
    """ModelTorchDistributed.gather_irregular() should raise NotImplementedError."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    tensor = torch.ones((2, 2), dtype=torch.float32, device=device)

    with pytest.raises(NotImplementedError, match="gather_irregular.*spatial"):
        dist.gather_irregular(tensor)


@requires_parallel
def test_gather_global_raises_under_spatial_parallelism(monkeypatch):
    """gather_global() delegates to gather(), so it should also raise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    tensor = torch.ones((2, 2), dtype=torch.float32, device=device)
    global_shape = (2, 2)

    with pytest.raises(NotImplementedError, match="gather.*spatial parallelism"):
        dist.gather_global(tensor, global_shape=global_shape, data_parallel_dim=0)
