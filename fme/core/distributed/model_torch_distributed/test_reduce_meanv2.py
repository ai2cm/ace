import logging

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed.utils import gather_helper_conv

logger = logging.getLogger(__name__)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
@pytest.mark.parametrize(
    "h_parallel,w_parallel",
    [
        (1, 2),
    ],
)
def test_reduce_mean_spatial_parallelism(h_parallel, w_parallel, monkeypatch):
    # Set up spatial parallelism
    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    dist = Distributed.get_instance()
    device = get_device()
    rank = dist.rank

    # Define dimensions
    n_lat = 1
    n_lon = 2

    # Create full tensor on host
    full_tensor = torch.tensor(
        [
            [[1.0, -1.0]],  # Batch 0
            [[2.0, -2.0]],  # Batch 1
        ]
    )

    logger.debug(f"Rank {rank}: Full tensor shape: {full_tensor.shape}")
    logger.debug(f"Rank {rank}: Full tensor:\n{full_tensor}")

    batch_ranks = dist.comm_get_size("data")

    # Get local slice for this rank
    local_slices = dist.get_local_slices((n_lat, n_lon))
    batch_slice = slice(rank // 2, rank // 2 + 1) if batch_ranks > 1 else slice(None)
    local_tensor = full_tensor[batch_slice, local_slices[0], local_slices[1]].clone()
    local_tensor = local_tensor.to(device)

    dist.barrier()

    logger.debug(f"Rank {rank}: batch_ranks {batch_ranks}, batch_slice: {batch_slice}")
    logger.debug(f"Rank {rank}: Local tensor before reduce: {local_tensor}")
    logger.debug(f"Rank {rank}: Local tensor shape: {local_tensor.shape}")

    # Reduce mean across batch dimension (data parallel group)
    result = dist.reduce_mean(local_tensor, group="data")

    logger.debug(f"Rank {rank}: Result after reduce_mean: {result}")

    # Gather across spatial dimensions
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Expected result: mean along batch dimension
    expected_mean = torch.mean(full_tensor, dim=0, keepdim=True)

    logger.debug(f"\nRank {rank}: Mean along dim=0 shape: {expected_mean.shape}")
    logger.debug(f"Rank {rank}: Mean along dim=0:\n{expected_mean}")
    logger.debug(f"\nRank {rank}: result_full shape: {result_full.shape}")
    logger.debug(f"Rank {rank}: result_full:\n{result_full}")

    # Move to CPU for comparison
    result_full_cpu = result_full.to("cpu")
    expected_mean_cpu = expected_mean.to("cpu")

    assert torch.allclose(result_full_cpu, expected_mean_cpu, rtol=1e-5, atol=1e-7), (
        f"Rank {rank}: Mismatch between result and expected.\n"
        f"Result: {result_full_cpu}\n"
        f"Expected: {expected_mean_cpu}"
    )
