import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed_utils import gather_helper_conv
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE, requires_parallel


@requires_parallel
@pytest.mark.parametrize(
    "h_parallel,w_parallel,nlat,nlon,nsamples",
    [
        (2, 1, 4, 8, 4),
        (1, 2, 4, 8, 4),
        (2, 1, 5, 8, 3),
        (1, 2, 4, 9, 3),
        (2, 2, 8, 9, 5),
        (3, 1, 9, 8, 2),
    ],
)
def test_get_local_slices(
    h_parallel,
    w_parallel,
    nlat,
    nlon,
    nsamples,
    monkeypatch,
):
    """Verify `get_local_slices` partitions spatial dims and `gather_helper_conv`
    reconstructs the original tensor.

    Parameterized over H/W decompositions and covers divisible and
    non-divisible domains.
    """

    if WORLD_SIZE % (h_parallel * w_parallel) != 0:
        pytest.skip(
            f"world_size={WORLD_SIZE} not divisible by "
            f"(h={h_parallel} * w={w_parallel})"
        )

    # Set up parallelization using monkeypatch
    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    # Create reference data on CPU
    torch.manual_seed(0)
    data_tensor_host = torch.randn(nsamples, nlat, nlon, device="cpu")

    # Get distributed instance and communication groups
    dist = Distributed.get_instance()
    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")

    # Get local slice of data
    this_shape = (nlat, nlon)
    tensor_data_local_host = (
        data_tensor_host[:, *dist.get_local_slices(this_shape)].detach().clone()
    )

    # Move to device
    device = get_device()
    tensor_data_local = tensor_data_local_host.to(device)

    # Gather data back to full tensor
    tensor_data_full = gather_helper_conv(
        tensor_data_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Move back to CPU for comparison
    tensor_data_full_cpu = tensor_data_full.to("cpu")

    # Verify reconstruction matches original
    torch.testing.assert_close(
        tensor_data_full_cpu,
        data_tensor_host,
        atol=0,
        rtol=0,
        msg=f"Gathered tensor does not match original "
        f"for H={h_parallel}, W={w_parallel}",
    )
