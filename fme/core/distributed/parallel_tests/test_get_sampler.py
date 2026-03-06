"""
Tests for Distributed.get_sampler.

Works with all backends (NonDistributed, TorchDistributed,
ModelTorchDistributed) and runs correctly both serially and under
``torchrun --nproc-per-node N``.

Key design notes
----------------
- gather_object collects over the data-parallel group (one entry per
  data-parallel rank), so spatial-parallel ranks that share the same
  data_parallel_rank do not inflate index counts.
- All dataset sizes that must be evenly split are multiples of
  total_data_parallel_ranks so no padding occurs unless explicitly tested.
"""

from typing import cast

import pytest
import torch

from fme.core.distributed import Distributed
from fme.core.rand import set_seed


def _make_dataset(n: int) -> torch.utils.data.TensorDataset:
    return torch.utils.data.TensorDataset(torch.arange(n, dtype=torch.float32))


@pytest.mark.parallel
def test_get_sampler_covers_all_indices():
    """
    With shuffle=False and a dataset evenly divisible by dp-ranks,
    the union of each rank's sampler indices covers every element
    exactly once.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    dataset = _make_dataset(4 * n_dp)

    sampler = dist.get_sampler(dataset, shuffle=False)
    local_indices = list(sampler)

    all_indices = dist.gather_object(local_indices)
    if dist.is_root():
        assert all_indices is not None
        gathered = cast(list[list[int]], all_indices)
        flat = [idx for rank_indices in gathered for idx in rank_indices]
        assert sorted(flat) == list(range(len(dataset)))


@pytest.mark.parallel
def test_get_sampler_drop_last_true():
    """
    With drop_last=True and a dataset whose size is not evenly divisible
    by the number of data-parallel ranks, the trailing remainder is
    discarded so every rank gets the same number of indices.

    When n_dp == 1 every size is evenly divisible, so drop_last is a
    no-op; we use a separate dataset size that still exercises the path.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks

    if n_dp == 1:
        # drop_last cannot actually drop anything with a single rank,
        # so just verify the sampler returns all indices unchanged.
        dataset = _make_dataset(3)
        sampler = dist.get_sampler(dataset, shuffle=False, drop_last=True)
        local_indices = list(sampler)
        assert len(local_indices) == 3
    else:
        # 3 * n_dp + 1 items -> floor-division gives 3 per rank,
        # the +1 remainder is dropped.
        dataset = _make_dataset(3 * n_dp + 1)
        sampler = dist.get_sampler(dataset, shuffle=False, drop_last=True)
        local_indices = list(sampler)
        assert len(local_indices) == 3

        all_indices = dist.gather_object(local_indices)
        if dist.is_root():
            assert all_indices is not None
            gathered = cast(list[list[int]], all_indices)
            flat = [idx for rank_indices in gathered for idx in rank_indices]
            # 3 * n_dp unique indices, all within dataset bounds, no duplicates
            assert len(flat) == 3 * n_dp
            assert len(set(flat)) == 3 * n_dp
            assert all(0 <= i < len(dataset) for i in flat)


@pytest.mark.parallel
def test_get_sampler_drop_last_false_pads():
    """
    With drop_last=False and a non-divisible dataset, DistributedSampler
    pads to the next multiple of n_dp so each rank gets the same count.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    # n_dp + 1 items → pads to 2 * n_dp, each rank gets 2
    dataset = _make_dataset(n_dp + 1)

    sampler = dist.get_sampler(dataset, shuffle=False, drop_last=False)
    assert len(list(sampler)) == 2


@pytest.mark.parallel
def test_get_sampler_seed_reproducibility():
    """
    Calling set_seed with the same value before each get_sampler call
    produces identical shuffled index sequences.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    dataset = _make_dataset(4 * n_dp)

    set_seed(42)
    first = list(dist.get_sampler(dataset, shuffle=True))

    set_seed(42)
    second = list(dist.get_sampler(dataset, shuffle=True))

    assert first == second
