import numpy as np
import pytest
import torch

from fme.core import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed import ModelTorchDistributed


@pytest.mark.parallel
def test_gather_tensor_from_local_slices():
    """
    Only tests get_local_slices and gather.

    Because the global data is the same on each rank, there is no coverage of
    "batch" parallelism in this test.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    global_shape = (2 * n_dp, 4, 4)
    x_global = (
        torch.arange(np.prod(global_shape), device=get_device()).reshape(global_shape)
        + 1
    )
    x_local = x_global[dist.get_local_slices(global_shape, data_parallel_dim=0)]
    gathered = dist.gather_global(
        x_local, global_shape=global_shape, data_parallel_dim=0
    )
    if dist.is_root():
        assert gathered is not None
        torch.testing.assert_close(gathered, x_global)
    else:
        assert gathered is None


@pytest.mark.parallel
def test_local_slices_cover_full_domain():
    """Union of slices from every rank should cover every element exactly once."""
    dist = Distributed.get_instance()
    if isinstance(dist._distributed, ModelTorchDistributed):
        pytest.xfail(
            "ModelTorchDistributed slicing along spatial dimensions is not implemented."
        )
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
def test_local_slices_subdivide_domain():
    """
    Only tests get_local_slices and gather.

    Tests that local slices subdivide the global domain into parts with no overlap
    that cover the entire domain.

    Because the global data is the same on each rank, there is no coverage of
    "batch" parallelism in this test.
    """
    dist = Distributed.get_instance()
    if isinstance(dist._distributed, ModelTorchDistributed):
        pytest.xfail(
            "ModelTorchDistributed slicing along spatial dimensions is not implemented."
        )
    n_dp = dist.total_data_parallel_ranks
    global_shape = (2 * n_dp, 4, 4)
    x_global = torch.zeros(global_shape, device=get_device())
    local_slices = dist.get_local_slices(global_shape, data_parallel_dim=0)
    gathered_local_slices = dist.gather_object(local_slices)
    if dist.is_root():
        assert gathered_local_slices is not None
        for i in range(n_dp):
            x_global[gathered_local_slices[i]] += 1.0
        torch.testing.assert_close(
            x_global, torch.ones_like(x_global)
        )  # the entire domain should get selected, and only once
    else:
        assert gathered_local_slices is None


@pytest.mark.parallel
def test_gather_global_tensor():
    """
    Test that gather_object and gather_global produce consistent results.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    global_shape = (2 * n_dp, 4, 4)
    x_global = torch.arange(np.prod(global_shape), device=get_device()).reshape(
        global_shape
    )
    local_slices = dist.get_local_slices(global_shape, data_parallel_dim=0)
    gathered_local_slices = dist.gather_object(local_slices)
    gathered_global = dist.gather_global(
        x_global[local_slices], global_shape=global_shape, data_parallel_dim=0
    )
    if dist.is_root():
        torch.testing.assert_close(gathered_global, x_global)
    else:
        assert gathered_local_slices is None


@pytest.mark.parallel
def test_local_slices_match_gather_tensors():
    """
    Test that gather_object and gather produce consistent results.
    """
    dist = Distributed.get_instance()
    n_dp = dist.total_data_parallel_ranks
    global_shape = (2 * n_dp, 4, 4)
    x_global = torch.arange(np.prod(global_shape), device=get_device()).reshape(
        global_shape
    )
    local_slices = dist.get_local_slices(global_shape, data_parallel_dim=0)
    gathered_local_slices = dist.gather_object(local_slices)
    gathered_tensors = dist.gather(
        x_global[local_slices],
    )
    if dist.is_root():
        assert gathered_local_slices is not None
        assert gathered_tensors is not None
        for i, (slices, tensor) in enumerate(
            zip(gathered_local_slices, gathered_tensors)
        ):
            torch.testing.assert_close(
                tensor,
                x_global[slices],
                msg=f"Rank {i} slices did not match gathered tensor",
            )
    else:
        assert gathered_local_slices is None
        assert gathered_tensors is None


@pytest.mark.parallel
def test_reduce_mean_from_multiple_ranks():
    """
    dist.reduce_mean should only reduce along the "data parallel" dimension, not
    along "model parallel" ranks.
    """
    dist = Distributed.get_instance()
    global_shape = (4, 4)
    x_global_base = torch.arange(
        global_shape[0] * global_shape[1], dtype=torch.float32, device=get_device()
    ).reshape(global_shape)
    # each global/model domain is a reshaped arange, with a different constant offset
    # depending on the batch/data parallel index/rank.
    x_global_ranked = x_global_base + dist.data_parallel_rank
    x_local_ranked = x_global_ranked[dist.get_local_slices(global_shape)]
    x_local_reduced = dist.reduce_mean(x_local_ranked)

    # we expect the offsets to average out, giving the arange map plus an average offset
    x_global_mean_expected = x_global_base + torch.mean(
        torch.arange(
            dist.total_data_parallel_ranks,
            dtype=x_global_base.dtype,
            device=x_global_base.device,
        )
    )
    # check the sub-domain we have on the local rank against this expectation
    x_local_reduced_expected = x_global_mean_expected[
        dist.get_local_slices(global_shape)
    ]
    torch.testing.assert_close(x_local_reduced, x_local_reduced_expected)
