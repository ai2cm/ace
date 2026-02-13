import logging

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed.utils import gather_helper_conv

logger = logging.getLogger(__name__)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires multi-GPU machine")
@pytest.mark.parametrize(
    "n_batch,n_lat,n_lon,h_parallel,w_parallel",
    [
        (2, 1, 2, 1, 2),
        (2, 2, 4, 2, 2),
    ],
)
def test_reduce_mean_spatial_parallelism(
    n_batch, n_lat, n_lon, h_parallel, w_parallel, monkeypatch
):
    """
    Test reduce_mean with spatial parallelism using random tensors.
    Parameters:
    - n_batch: Number of samples in batch dimension
    - n_lat: Number of latitude points
    - n_lon: Number of longitude points
    - h_parallel: Split size for latitude dimension (H_PARALLEL_SIZE)
    - w_parallel: Split size for longitude dimension (W_PARALLEL_SIZE)

    The test verifies that reduce_mean correctly averages across the batch dimension
    when using spatial parallelism.
    """
    # Set up spatial parallelism
    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    dist = Distributed.get_instance()
    device = get_device()
    rank = dist.rank
    batch_ranks = dist.comm_get_size("data")

    # Create full tensor with random values
    # Use fixed seed for reproducibility across all ranks
    torch.manual_seed(0)
    full_tensor = torch.randn((n_batch, n_lat, n_lon))

    if rank == 0:
        logger.debug(f"\n{'='*60}")
        logger.debug(f"Test configuration:")
        logger.debug(f"  Batch size: {n_batch}")
        logger.debug(f"  Lat size: {n_lat}")
        logger.debug(f"  Lon size: {n_lon}")
        logger.debug(f"  H_PARALLEL_SIZE: {h_parallel}")
        logger.debug(f"  W_PARALLEL_SIZE: {w_parallel}")
        logger.debug(f"{'='*60}\n")
        logger.debug(f"Full tensor shape: {full_tensor.shape}")
        logger.debug(f"Full tensor:\n{full_tensor}")

    # Get local slice for this rank
    local_slices = dist.get_local_slices((n_lat, n_lon))

    # Calculate which batch slice this rank should have
    batches_per_rank = n_batch // batch_ranks
    batch_start = (rank // w_parallel) * batches_per_rank
    batch_end = batch_start + batches_per_rank
    batch_slice = slice(batch_start, batch_end) if batch_ranks > 1 else slice(None)

    # Extract local tensor
    local_tensor = full_tensor[batch_slice, local_slices[0], local_slices[1]].clone()
    local_tensor = local_tensor.to(device)

    dist.barrier()

    logger.debug(f"\nRank {rank}:")
    logger.debug(f"  Batch ranks: {batch_ranks}")
    logger.debug(f"  Batch slice: {batch_slice}")
    logger.debug(f"  Local slices (lat, lon): {local_slices}")
    logger.debug(f"  Local tensor shape: {local_tensor.shape}")
    logger.debug(f"  Local tensor:\n{local_tensor}")

    # Reduce mean across batch dimension
    # NOTE: here we pass "data" as group name.
    result = dist.reduce_mean(local_tensor, group="data")

    logger.debug(f"  Result after reduce_mean shape: {result.shape}")
    logger.debug(f"  Result after reduce_mean:\n{result}")

    # Gather results from all ranks
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Calculate expected mean
    expected_mean = torch.mean(full_tensor, dim=0, keepdim=True)

    if rank == 0:
        logger.debug(f"\n{'='*60}")
        logger.debug(f"Expected mean shape: {expected_mean.shape}")
        logger.debug(f"Expected mean:\n{expected_mean}")
        logger.debug(f"\nGathered result shape: {result_full.shape}")
        logger.debug(f"Gathered result:\n{result_full}")
        logger.debug(f"{'='*60}\n")

    # Verify correctness
    result_cpu = result_full.to("cpu")
    expected_cpu = expected_mean.to("cpu")

    # Use torch.allclose for floating point comparison
    assert torch.allclose(
        result_cpu, expected_cpu, rtol=1e-5, atol=1e-7
    ), f"Rank {rank}: Mismatch!\nExpected:\n{expected_cpu}\nGot:\n{result_cpu}"
