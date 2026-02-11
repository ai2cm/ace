import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed.utils import gather_helper_conv


@pytest.mark.parametrize(
    "n_batch,n_lat,n_lon,h_parallel,w_parallel",
    [
        (2, 1, 2, 1, 2),
        (2, 2, 4, 2, 2),
    ],
)
def test_reduce_mean_spatial_parallelism(n_batch, n_lat, n_lon, h_parallel, w_parallel):
    """
    Test reduce_mean with spatial parallelism using random tensors.

    Parameters:
    - n_batch: Number of samples in batch dimension
    - n_lat: Number of latitude points
    - n_lon: Number of longitude points
    - h_parallel: Split size for batch dimension (H_PARALLEL_SIZE)
    - w_parallel: Split size for longitude dimension (W_PARALLEL_SIZE)
    - seed: Random seed for reproducibility

    The test verifies that reduce_mean correctly averages across the batch dimension
    when using spatial parallelism.
    """
    # Set up spatial parallelism
    os.environ["H_PARALLEL_SIZE"] = str(h_parallel)
    os.environ["W_PARALLEL_SIZE"] = str(w_parallel)

    dist = Distributed.get_instance()
    device = get_device()
    rank = dist.rank
    batch_ranks = dist.comm_get_size("data")

    # Create full tensor with random values
    # Use fixed seed for reproducibility across all ranks
    torch.manual_seed(0)

    full_tensor = torch.randn((n_batch, n_lat, n_lon))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Test configuration:")
        print(f"  Batch size: {n_batch}")
        print(f"  Lat size: {n_lat}")
        print(f"  Lon size: {n_lon}")
        print(f"  H_PARALLEL_SIZE: {h_parallel}")
        print(f"  W_PARALLEL_SIZE: {w_parallel}")
        print(f"{'='*60}\n")
        print(f"Full tensor shape: {full_tensor.shape}")
        print(f"Full tensor:\n{full_tensor}")

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

    print(f"\nRank {rank}:")
    print(f"  Batch ranks: {batch_ranks}")
    print(f"  Batch slice: {batch_slice}")
    print(f"  Local slices (lat, lon): {local_slices}")
    print(f"  Local tensor shape: {local_tensor.shape}")
    print(f"  Local tensor:\n{local_tensor}")

    # Reduce mean across batch dimension
    result = dist.reduce_mean(local_tensor)

    print(f"  Result after reduce_mean shape: {result.shape}")
    print(f"  Result after reduce_mean:\n{result}")

    # Gather results from all ranks
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Calculate expected mean
    expected_mean = torch.mean(full_tensor, dim=0, keepdim=True)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Expected mean shape: {expected_mean.shape}")
        print(f"Expected mean:\n{expected_mean}")
        print(f"\nGathered result shape: {result_full.shape}")
        print(f"Gathered result:\n{result_full}")
        print(f"{'='*60}\n")

    # Verify correctness
    result_cpu = result_full.to("cpu")
    expected_cpu = expected_mean.to("cpu")

    # Clean up environment variables
    os.environ.pop("H_PARALLEL_SIZE", None)
    os.environ.pop("W_PARALLEL_SIZE", None)

    # Use torch.allclose for floating point comparison
    assert torch.allclose(
        result_cpu, expected_cpu, rtol=1e-5, atol=1e-7
    ), f"Rank {rank}: Mismatch!\nExpected:\n{expected_cpu}\nGot:\n{result_cpu}"
