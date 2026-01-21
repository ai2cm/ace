"""
Tests for local_batch_size under parallelism modes
"""

import pytest

from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE, requires_parallel


@requires_parallel
def test_local_batch_size_spatial_parallelism(monkeypatch):
    """
    Under full spatial parallelism (all ranks used for model parallelism),
    data parallel group size is 1, so local_batch_size == global batch_size.
    """
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(WORLD_SIZE))

    dist = Distributed.get_instance()

    # All ranks are used for spatial parallelism, data group size = 1
    assert dist.total_data_parallel_ranks == 1
    assert dist.local_batch_size(16) == 16


@requires_parallel
def test_local_batch_size_mixed_parallelism(monkeypatch):
    """
    With partial spatial parallelism, some ranks are data-parallel.
    local_batch_size should divide by data parallel group size.

    Requires world_size >= 2. If world_size == 2: W=2 means data_group=1
    (same as full spatial). This test is most meaningful at world_size >= 4,
    but still passes at 2 since data_group=1 there.
    """

    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", "2")

    if WORLD_SIZE % 2 != 0:
        pytest.skip(f"world_size={WORLD_SIZE} not divisible by spatial_size=2")

    dist = Distributed.get_instance()

    global_batch = 32
    data_group_size = dist.total_data_parallel_ranks
    assert data_group_size >= 1
    expected = global_batch // data_group_size
    assert dist.local_batch_size(global_batch) == expected


def test_local_batch_size_not_divisible():
    """When batch_size is not divisible by data group size, integer division
    truncates. This documents the expected (lossy) behavior."""
    with Distributed.force_non_distributed():
        dist = Distributed.get_instance()
        # NonDistributed has data group size of 1, so any batch goes through.
        assert dist.local_batch_size(7) == 7

    # Simulate a distributed backend with 2 data-parallel ranks to verify truncation
    class FakeBackend:
        def __init__(self, total_data_parallel_ranks):
            self.total_data_parallel_ranks = total_data_parallel_ranks

        def local_batch_size(self, batch_size: int) -> int:
            return batch_size // self.total_data_parallel_ranks

    fake = FakeBackend(total_data_parallel_ranks=2)
    # from fme.core.distributed import Distributed as _Distributed

    with Distributed.replace_backend(fake):  # type: ignore[arg-type]
        dist = Distributed.get_instance()
        assert dist.local_batch_size(7) == 3
