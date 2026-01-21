import logging

import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed_utils import gather_helper_conv
from fme.core.distributed.parallel_tests._helpers import (
    WORLD_SIZE,
    requires_parallel,
    requires_world_size,
)

logger = logging.getLogger(__name__)

# Floating point tolerances for assertions
RTOL = 1e-5
ATOL = 1e-7


# Notes:
# - Process groups for spatial/data/world are initialized once per torchrun
#   session by the backend. Setting `H_PARALLEL_SIZE`/`W_PARALLEL_SIZE` via
#   `monkeypatch` in tests attempts to influence that initialization but the
#   actual group sizes come from the runtime environment. Tests compute
#   expected values using the runtime group sizes and skip when the chosen
#   synthetic parameters are incompatible with `WORLD_SIZE`.


def _expected_reduce_mean_from_full(full_tensor: torch.Tensor, batch_ranks: int):
    """Compute expected reduce-mean over the data-parallel (batch) axis.

    Mirrors the logic used in tests: when `batch_ranks > 1`, the full tensor is
    split into `batch_ranks` consecutive chunks along dim=0 and averaged.
    """
    if batch_ranks > 1:
        batches_per_rank = full_tensor.shape[0] // batch_ranks
        chunks = [
            full_tensor[i * batches_per_rank : (i + 1) * batches_per_rank]
            for i in range(batch_ranks)
        ]
        return torch.stack(chunks).mean(dim=0)
    return full_tensor


@pytest.fixture
def make_dist():
    """Factory fixture that returns a fresh Distributed instance and groups.

    This is a factory (callable) because some tests set env vars via
    `monkeypatch.setenv(...)` in the test body before creating the
    Distributed instance.
    """

    def _make():
        dist = Distributed.get_instance()
        device = get_device()
        w_group = dist.comm_get_group("w")
        h_group = dist.comm_get_group("h")
        return dist, device, w_group, h_group

    return _make


@requires_world_size(4)
@pytest.mark.parametrize(
    "n_batch,n_lat,n_lon,h_parallel,w_parallel",
    [
        (2, 1, 2, 1, 2),
        (2, 2, 4, 2, 2),
    ],
)
def test_reduce_mean_sp4(
    n_batch,
    n_lat,
    n_lon,
    h_parallel,
    w_parallel,
    make_dist,
    monkeypatch,
):
    """
    Test reduce_mean with spatial parallelism using random tensors.

    NOTE: physicsnemo comm groups are initialized once per torchrun session
    and cannot be re-initialized, so the actual data/spatial group sizes may
    differ from h_parallel/w_parallel set via monkeypatch. The expected value
    is computed dynamically using the actual group sizes from the backend.
    """
    if WORLD_SIZE % (h_parallel * w_parallel) != 0:
        pytest.skip(
            f"world_size={WORLD_SIZE} not divisible by "
            f"(h={h_parallel} * w={w_parallel})"
        )
    # Set up spatial parallelism
    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    dist, device, w_group, h_group = make_dist()
    batch_ranks = dist.comm_get_size("data")
    batch_rank = dist.comm_get_rank("data")

    # Create full tensor with random values
    # Use fixed seed for reproducibility across all ranks
    torch.manual_seed(0)
    full_tensor = torch.randn((n_batch, n_lat, n_lon))

    # Get local slice for this rank
    local_slices = dist.get_local_slices((n_lat, n_lon))

    # Calculate which batch slice this rank should have
    batches_per_rank = n_batch // batch_ranks if batch_ranks > 1 else n_batch
    batch_start = batch_rank * batches_per_rank
    batch_end = batch_start + batches_per_rank
    batch_slice = slice(batch_start, batch_end) if batch_ranks > 1 else slice(None)

    # Extract local tensor
    local_tensor = full_tensor[batch_slice, local_slices[0], local_slices[1]].clone()
    local_tensor = local_tensor.to(device)

    dist.barrier()

    # Reduce mean across batch dimension
    # NOTE: here we pass "data" as group name.
    result = dist.reduce_mean(local_tensor, group="data")

    # Gather results from all ranks
    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Calculate expected result based on actual data parallelism.
    # reduce_mean on "data" averages element-wise across data-parallel ranks.
    # When data_group_size > 1, each rank has a batch subset; after averaging,
    # the result is the mean across all batch subsets.
    # When data_group_size == 1, reduce_mean is a no-op.
    if batch_ranks > 1:
        # Each data rank contributed batches_per_rank consecutive batches.
        # After all_reduce AVG, the result is the mean across all D subsets.
        chunks = [
            full_tensor[i * batches_per_rank : (i + 1) * batches_per_rank]
            for i in range(batch_ranks)
        ]
        expected = torch.stack(chunks).mean(dim=0)
    else:
        expected = full_tensor

    # Verify correctness
    result_cpu = result_full.to("cpu")
    expected_cpu = expected.to("cpu")

    # Use torch.allclose for floating point comparison
    torch.testing.assert_close(result_cpu, expected_cpu, rtol=RTOL, atol=ATOL)


@requires_world_size(2)
@pytest.mark.parametrize(
    "h_parallel,w_parallel",
    [
        (1, 2),
    ],
)
def test_reduce_mean_sp2(
    h_parallel,
    w_parallel,
    make_dist,
    monkeypatch,
):
    if WORLD_SIZE % (h_parallel * w_parallel) != 0:
        pytest.skip(
            f"world_size={WORLD_SIZE} not divisible by "
            f"(h={h_parallel} * w={w_parallel})"
        )

    # Set up spatial parallelism
    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    dist, device, w_group, h_group = make_dist()

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

    batch_ranks = dist.comm_get_size("data")
    batch_rank = dist.comm_get_rank("data")

    # Get local slice for this rank
    local_slices = dist.get_local_slices((n_lat, n_lon))
    batch_slice = slice(batch_rank, batch_rank + 1) if batch_ranks > 1 else slice(None)
    local_tensor = full_tensor[batch_slice, local_slices[0], local_slices[1]].clone()
    local_tensor = local_tensor.to(device)

    dist.barrier()

    # Under 2-way spatial split (W=2), data group has size 1, so this is a no-op.
    result = dist.reduce_mean(local_tensor, group="data")

    # Gather across spatial dimensions
    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    # Expected result: unchanged local tensors gathered across spatial dimensions.
    expected_mean = full_tensor

    # Move to CPU for comparison
    result_full_cpu = result_full.to("cpu")
    expected_mean_cpu = expected_mean.to("cpu")

    torch.testing.assert_close(result_full_cpu, expected_mean_cpu, rtol=RTOL, atol=ATOL)


@requires_world_size(4)
def test_reduce_mean_non_square_split(make_dist, monkeypatch):
    """Edge-case test: non-square spatial split (H != W).

    Uses a small, deterministic full tensor so the gathered result and the
    expected value are easy to verify. The expected value is computed based
    on the runtime data-group size.
    """
    h_parallel = 2
    w_parallel = 1

    if WORLD_SIZE % (h_parallel * w_parallel) != 0:
        pytest.skip(
            f"world_size={WORLD_SIZE} not divisible by "
            f"(h={h_parallel} * w={w_parallel})"
        )

    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    dist, device, w_group, h_group = make_dist()

    # Small deterministic tensor: shape (2, 2, 4)
    n_batch = 2
    n_lat = 2
    n_lon = 4
    full_tensor = torch.arange(n_batch * n_lat * n_lon, dtype=torch.float32).reshape(
        n_batch, n_lat, n_lon
    )

    batch_ranks = dist.comm_get_size("data")
    batch_rank = dist.comm_get_rank("data")

    local_slices = dist.get_local_slices((n_lat, n_lon))
    batches_per_rank = n_batch // batch_ranks if batch_ranks > 1 else n_batch
    batch_slice = (
        slice(
            batch_rank * batches_per_rank,
            batch_rank * batches_per_rank + batches_per_rank,
        )
        if batch_ranks > 1
        else slice(None)
    )

    local_tensor = full_tensor[batch_slice, local_slices[0], local_slices[1]].clone()
    local_tensor = local_tensor.to(device)

    dist.barrier()

    result = dist.reduce_mean(local_tensor, group="data")

    result_full = gather_helper_conv(
        result, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )

    expected = _expected_reduce_mean_from_full(full_tensor, batch_ranks)

    torch.testing.assert_close(
        result_full.to("cpu"), expected.to("cpu"), rtol=RTOL, atol=ATOL
    )


@requires_world_size(2)
def test_reduce_mean_world_group_two_gpu(monkeypatch):
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", "2")

    dist = Distributed.get_instance()
    device = get_device()

    rank = dist.rank
    assert dist.comm_get_size("data") == 1

    full_tensor = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    local_tensor = full_tensor[rank : rank + 1].clone().to(device)

    dist.barrier()

    result = dist.reduce_mean(local_tensor, group="world")
    expected_mean = torch.mean(full_tensor, dim=0, keepdim=True).to("cpu")

    torch.testing.assert_close(result.to("cpu"), expected_mean)


@requires_parallel
def test_reduce_mean_world_group_any_gpu_count(monkeypatch):
    world_size = WORLD_SIZE
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()
    rank = dist.rank

    n_lat = 2
    n_lon = world_size * 2
    local_slices = dist.get_local_slices((n_lat, n_lon))
    local_h = local_slices[0].stop - local_slices[0].start
    local_w = local_slices[1].stop - local_slices[1].start

    local_tensor = torch.full(
        (1, local_h, local_w),
        fill_value=float(rank + 1),
        dtype=torch.float32,
        device=device,
    )

    dist.barrier()

    result_local = dist.reduce_mean(local_tensor, group="world")
    expected_value = torch.tensor(
        (world_size + 1) / 2.0, dtype=torch.float32, device=device
    )
    torch.testing.assert_close(
        result_local, torch.full_like(result_local, expected_value)
    )

    w_group = dist.comm_get_group("w")
    h_group = dist.comm_get_group("h")
    result_full = gather_helper_conv(
        result_local, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group
    )
    expected_full = torch.full((1, n_lat, n_lon), expected_value, device=device)
    torch.testing.assert_close(result_full, expected_full)


@requires_parallel
def test_reduce_mean_default_group(monkeypatch):
    """
    Test reduce_mean with group=None (default process group) under
    spatial parallelism. This exercises the ModelTorchDistributed path
    where comm.get_group(None) returns None → uses the default group.
    """
    world_size = WORLD_SIZE
    monkeypatch.setenv("H_PARALLEL_SIZE", "1")
    monkeypatch.setenv("W_PARALLEL_SIZE", str(world_size))

    dist = Distributed.get_instance()
    device = get_device()

    # Each rank contributes its rank index
    local_tensor = torch.full(
        (2, 2), fill_value=float(dist.rank), dtype=torch.float32, device=device
    )

    # reduce_mean with default group averages across all ranks
    result = dist.reduce_mean(local_tensor)

    expected_value = sum(range(world_size)) / world_size
    expected = torch.full_like(result, expected_value)
    torch.testing.assert_close(result, expected)
