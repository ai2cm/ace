"""
Backend-agnostic distributed tests.

These tests work with any backend (NonDistributed, TorchDistributed,
ModelTorchDistributed) and on CPU or GPU.  They can be run serially
(``pytest``) or in parallel (``torchrun --nproc-per-node N -m pytest``).

Every test constructs payloads whose expected result can be computed
analytically, so the outcome is deterministic regardless of rank count.
"""

import pytest
import torch

from fme.core import get_device
from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_reduce_mean_all_ones():
    """Reducing an all-ones tensor should return all ones."""
    dist = Distributed.get_instance()
    t = torch.ones(4, 4, device=get_device())
    result = dist.reduce_mean(t)
    torch.testing.assert_close(result, torch.ones_like(t))


@pytest.mark.parallel
def test_reduce_sum_all_ones():
    """Sum of all-ones across N dp-ranks should give N."""
    dist = Distributed.get_instance()
    t = torch.ones(3, device=get_device())
    result = dist.reduce_sum(t)
    expected = torch.full_like(t, dist.total_data_parallel_ranks)
    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_reduce_min_constant():
    """Min of a constant tensor is the same constant."""
    dist = Distributed.get_instance()
    t = torch.full((4,), 7.0, device=get_device())
    result = dist.reduce_min(t)
    torch.testing.assert_close(result, t.clone())


@pytest.mark.parallel
def test_reduce_max_constant():
    """Max of a constant tensor is the same constant."""
    dist = Distributed.get_instance()
    t = torch.full((4,), 3.0, device=get_device())
    result = dist.reduce_max(t)
    torch.testing.assert_close(result, t.clone())


@pytest.mark.parallel
def test_reduce_mean_rank_offset():
    """Each dp-rank adds its rank index.  Mean should equal base + avg(ranks)."""
    dist = Distributed.get_instance()
    base = torch.arange(6, dtype=torch.float32, device=get_device())
    t = base + dist.data_parallel_rank
    result = dist.reduce_mean(t)
    avg_offset = (dist.total_data_parallel_ranks - 1) / 2.0
    expected = base + avg_offset
    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_reduce_mean_data_parallel_group_only():
    """reduce_mean must average only over data-parallel ranks, not all ranks.

    Each rank sets its tensor value to its global rank.  Because global ranks
    differ *within* a data-parallel group (across spatial/h/w sub-ranks), the
    group mean differs from the all-rank mean whenever spatial_size > 1.  A
    buggy implementation that reduces over all ranks would produce the wrong
    answer in that case.

    For a mesh shaped (data, h, w) in row-major order the global ranks in a
    data-parallel group share the same spatial offset:
        spatial_offset = rank - data_parallel_rank * spatial_size
    and the expected group mean is:
        spatial_offset + spatial_size * (total_data_parallel_ranks - 1) / 2
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    spatial_size = dist.world_size // n_dp

    t = torch.full((1,), float(dist.rank), device=get_device())
    result = dist.reduce_mean(t)

    spatial_offset = dist.rank - dist.data_parallel_rank * spatial_size
    expected_val = spatial_offset + spatial_size * (n_dp - 1) / 2.0
    torch.testing.assert_close(result, torch.full_like(result, expected_val))


@pytest.mark.parallel
def test_reduce_min_selects_smallest_rank():
    """Each dp-rank holds (base + rank).  Min should select rank-0 values."""
    dist = Distributed.get_instance()
    base = torch.arange(4, dtype=torch.float32, device=get_device())
    t = base + dist.data_parallel_rank
    result = dist.reduce_min(t)
    expected = base  # rank-0 values are the smallest
    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_reduce_max_selects_largest_rank():
    """Each dp-rank holds (base + rank).  Max should select last-rank values."""
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    base = torch.arange(4, dtype=torch.float32, device=get_device())
    t = base + dist.data_parallel_rank
    result = dist.reduce_max(t)
    expected = base + (n_dp - 1)
    torch.testing.assert_close(result, expected)


@pytest.mark.parallel
def test_gather_produces_correct_count():
    """Root should receive one tensor per dp-rank; others get None."""
    dist = Distributed.get_instance()
    t = torch.full((2, 3), float(dist.data_parallel_rank), device=get_device())
    gathered = dist.gather(t)
    if dist.is_root():
        assert gathered is not None
        assert len(gathered) == dist.total_data_parallel_ranks
    else:
        assert gathered is None


@pytest.mark.parallel
def test_gather_irregular_matching_shapes():
    """gather_irregular with identical shapes works like gather."""
    dist = Distributed.get_instance()
    t = torch.full((4,), float(dist.data_parallel_rank), device=get_device())
    gathered = dist.gather_irregular(t)
    if dist.is_root():
        assert gathered is not None
        assert len(gathered) == dist.total_data_parallel_ranks
        for i, g in enumerate(gathered):
            torch.testing.assert_close(
                g.to(get_device()),
                torch.full((4,), float(i), device=get_device()),
            )
    else:
        assert gathered is None


@pytest.mark.parallel
def test_gather_global_reconstructs_arange():
    """
    Split an arange tensor across dp-ranks and verify that
    gather_global reconstructs the original.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    batch = 4 * n_dp  # evenly divisible
    global_shape = (batch, 3)
    x_global = torch.arange(
        batch * 3, dtype=torch.float32, device=get_device()
    ).reshape(global_shape)
    x_local = x_global[dist.get_local_slices(global_shape, data_parallel_dim=0)]
    reconstructed = dist.gather_global(
        x_local, global_shape=global_shape, data_parallel_dim=0
    )
    if dist.is_root():
        assert reconstructed is not None
        torch.testing.assert_close(reconstructed, x_global)
    else:
        assert reconstructed is None


@pytest.mark.parallel
def test_local_batch_size_divisibility():
    """local_batch_size * dp_ranks == global_batch_size."""
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    global_bs = 16 * n_dp  # always evenly divisible
    local_bs = dist.local_batch_size(global_bs)
    assert local_bs * n_dp == global_bs


@pytest.mark.parallel
def test_local_slices_cover_full_domain():
    """Union of slices from every rank should cover every element exactly once."""
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    rows = 4 * n_dp
    global_shape = (rows, 6)
    local_slices = dist.get_local_slices(global_shape, data_parallel_dim=0)
    # Collect one slice per data-parallel rank on root and verify full coverage.
    # gather_object gathers over the data-parallel group, so all_slices has
    # n_dp entries — one unique slice per dp rank, each covering a distinct
    # portion of the domain.
    all_slices = dist.gather_object(local_slices)
    if dist.is_root():
        assert all_slices is not None
        canvas = torch.zeros(global_shape)
        for s in all_slices:
            canvas[s] += 1
        torch.testing.assert_close(canvas, torch.ones_like(canvas))


@pytest.mark.parallel
def test_local_slices_no_dp_dim():
    """Without a dp dim, every rank gets full slices."""
    dist = Distributed.get_instance()
    slices = dist.get_local_slices((8, 4))
    assert slices == tuple(slice(None, None) for _ in range(2))


@pytest.mark.parallel
def test_wrap_module_preserves_underlying():
    """Wrapped module should expose .module pointing at the original."""
    dist = Distributed.get_instance()
    mod = torch.nn.Linear(4, 4).to(get_device())
    wrapped = dist.wrap_module(mod)
    assert hasattr(wrapped, "module")
    assert wrapped.module is mod


@pytest.mark.parallel
def test_barrier_does_not_hang():
    """Barrier should complete without deadlock."""
    dist = Distributed.get_instance()
    dist.barrier()  # simply verify it returns


@pytest.mark.parallel
def test_rank_within_world_size():
    dist = Distributed.get_instance()
    assert 0 <= dist.rank < dist.world_size


@pytest.mark.parallel
def test_data_parallel_rank_within_total():
    dist = Distributed.get_instance()
    assert 0 <= dist.data_parallel_rank < dist.total_data_parallel_ranks


@pytest.mark.parallel
def test_world_size_positive():
    dist = Distributed.get_instance()
    assert dist.world_size >= 1
