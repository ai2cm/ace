import pytest
import torch

from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed.utils import gather_helper_conv


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
@pytest.mark.parametrize(
    "h_parallel,w_parallel",
    [
        (2, 1),  # H-parallel split
        (1, 2),  # W-parallel split
    ],
)
def test_get_local_slices(h_parallel, w_parallel, monkeypatch):
    """Test that get_local_slices correctly
    distributes data and gather reconstructs it."""

    # Set up parallelization using monkeypatch
    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    # Define tensor dimensions
    nsamples = 4
    nlat = 4
    nlot = 8

    # Create reference data on CPU
    torch.manual_seed(0)
    data_tensor_host = torch.randn(nsamples, nlat, nlot, device="cpu")

    # Get distributed instance and communication groups
    dist = Distributed.get_instance()
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")

    # Get local slice of data
    this_shape = (nlat, nlot)
    tensor_data_local_host = (
        data_tensor_host[:, *dist.get_local_slices(this_shape)].detach().clone()
    )

    # Move to GPU
    tensor_data_local = tensor_data_local_host.to(dist.get_local_rank())

    # Gather data back to full tensor
    tensor_data_full = gather_helper_conv(
        tensor_data_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Move back to CPU for comparison
    tensor_data_full_cpu = tensor_data_full.to("cpu")

    # Verify reconstruction matches original
    assert torch.equal(
        tensor_data_full_cpu, data_tensor_host
    ), f"Gathered tensor does not match original for H={h_parallel}, W={w_parallel}"
