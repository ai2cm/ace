import os

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed.utils import gather_helper_conv


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
def test_reduce_mean_spatial_parallelism():
    # Set up spatial parallelism
    os.environ["H_PARALLEL_SIZE"] = "1"  # Split batch
    os.environ["W_PARALLEL_SIZE"] = "2"  # Split longitude

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

    print(f"Rank {rank}: Full tensor shape: {full_tensor.shape}")
    print(f"Rank {rank}: Full tensor:\n{full_tensor}")
    batch_ranks = dist.comm_get_size("data")

    # Get local slice for this rank
    local_slices = dist.get_local_slices((n_lat, n_lon))
    batch_slice = slice(rank // 2, rank // 2 + 1) if batch_ranks > 1 else slice(None)

    local_tensor = full_tensor[batch_slice, local_slices[0], local_slices[1]].clone()
    local_tensor = local_tensor.to(device)
    dist.barrier()

    print(f" batch_ranks {batch_ranks} batch_slice : {batch_slice}")

    print(f"Rank {rank}: Local tensor before reduce: {local_tensor}")
    print(f"Rank {rank}: Local tensor shape: {local_tensor.shape}")

    # Reduce mean across batch dimension (H parallel group)
    result = dist.reduce_mean(local_tensor)  #

    print(f"Rank {rank}: Result after reduce_mean: {result}")

    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")

    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    expected_mean = torch.mean(full_tensor, dim=0, keepdim=True)

    print("\nMean along dim=0 shape:", expected_mean.shape)  # torch.Size([2, 2])
    print("Mean along dim=0:\n", expected_mean)

    print("\nresult_fullshape:", result_full.shape)  # torch.Size([2, 2])
    print("result_full :\n", result_full)

    assert torch.equal(result_full.to("cpu"), expected_mean.to("cpu"))
