import numpy as np
import torch

from fme.core import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE


def test_gather_tensor_from_local_slices():
    """
    Only tests get_local_slices and gather.

    Because the global data is the same on each rank, there is no coverage of
    "batch" parallelism in this test.
    """
    dist = Distributed.get_instance()
    global_shape = (WORLD_SIZE * 2, 4, 4)
    x_global = (
        torch.arange(np.prod(global_shape), device=get_device()).reshape(global_shape)
        + 1
    )
    x_local = x_global[
        dist.get_local_slices(global_shape, dist.rank, data_parallel_dim=0)
    ]
    gathered = dist.gather_global(
        x_local, global_shape=global_shape, data_parallel_dim=0
    )
    if dist.is_root():
        assert gathered is not None
        torch.testing.assert_close(gathered, x_global)
    else:
        assert gathered is None


def test_local_slices_subdivide_domain():
    """
    Only tests get_local_slices and gather.

    Because the global data is the same on each rank, there is no coverage of
    "batch" parallelism in this test.
    """
    dist = Distributed.get_instance()
    global_shape = (WORLD_SIZE * 2, 4, 4)
    x_global = torch.zeros(global_shape, device=get_device())
    total_size = np.prod(global_shape)
    assert (
        total_size % dist.world_size == 0
    ), "total_size is not divisible by total ranks"
    expected_slice_size = total_size // dist.world_size
    for i in range(dist.world_size):
        local_slices = dist.get_local_slices(global_shape, i, data_parallel_dim=0)
        # the slices should be of the minimum size required
        assert x_global[local_slices].nelement() == expected_slice_size
        x_global[local_slices] = 1
    torch.testing.assert_close(
        x_global, torch.ones_like(x_global)
    )  # the entire domain should get selected


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
    x_local_ranked = x_global_ranked[dist.get_local_slices(global_shape, dist.rank)]
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
        dist.get_local_slices(global_shape, dist.rank)
    ]
    torch.testing.assert_close(x_local_reduced, x_local_reduced_expected)
