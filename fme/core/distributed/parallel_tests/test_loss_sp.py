import numpy as np
import pytest
import torch

from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE, requires_parallel
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import LossConfig


@requires_parallel
@pytest.mark.parametrize("global_mean_type", [None])
@pytest.mark.parametrize(
    "h_parallel,w_parallel",
    [
        (2, 1),  # H-parallel split
        (1, 2),  # W-parallel split
    ],
)
def test_loss_builds_and_runs_with_spatial_parallelism(
    global_mean_type, h_parallel, w_parallel, monkeypatch
):
    """Test that loss computation is consistent between
    serial and parallel execution."""
    world_size = WORLD_SIZE
    spatial_size = h_parallel * w_parallel
    if world_size % spatial_size != 0:
        pytest.skip(
            f"world_size={world_size} not divisible by spatial_size={spatial_size}"
        )
    error_tol = 1e-7
    nx = 8
    ny = 8

    torch.manual_seed(0)

    # Create test data
    data_tensor_host = torch.randn(1, 2, nx, ny, device="cpu")
    area_weights_host = torch.ones(nx, ny).to("cpu") * 5

    x_host = torch.randn(1, 2, nx, ny, device="cpu")
    y_host = torch.randn(1, 2, nx, ny, device="cpu")
    example_data_host = {"a": data_tensor_host}

    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    # Use the same data from serial execution
    area_weights = area_weights_host.to(get_device()) * 1.0  # Create a copy

    # Get distributed instance and prepare local data
    dist = Distributed.get_instance()

    # Partition data tensor across spatial dimensions
    this_shape = (data_tensor_host.shape[-2], data_tensor_host.shape[-1])
    tensor_data_local_host = (
        (data_tensor_host[:, :, *dist.get_local_slices(this_shape)]).detach().clone()
    )
    tensor_data_local = tensor_data_local_host.to(dist.get_local_rank())
    example_data_local = {"a": tensor_data_local}

    # Partition x and y tensors
    this_shape_x = (x_host.shape[-2], x_host.shape[-1])
    x_local_host = (x_host[:, :, *dist.get_local_slices(this_shape_x)]).detach().clone()
    x_local = x_local_host.to(dist.get_local_rank())

    y_local_host = (y_host[:, :, *dist.get_local_slices(this_shape_x)]).detach().clone()
    y_local = y_local_host.to(dist.get_local_rank())

    # Build loss function (same config as serial)
    config = LossConfig(global_mean_type=global_mean_type)
    loss_fn_parallel = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area_weights),
    )

    # Compute loss on local partitions
    result_local = loss_fn_parallel(x_local, y_local)

    # Aggregate and get logs
    aggregator_parallel = MeanAggregator(LatLonOperations(area_weights))
    aggregator_parallel.record_batch(
        loss=result_local,
        target_data=example_data_local,
        gen_data=example_data_local,
        target_data_norm=example_data_local,
        gen_data_norm=example_data_local,
    )
    logs_parallel = aggregator_parallel.get_logs(label="metrics")
    loss_parallel = logs_parallel["metrics/loss"]

    torch.manual_seed(0)
    with Distributed.force_non_distributed():
        # Build loss function
        config = LossConfig(global_mean_type=global_mean_type)
        loss_fn = config.build(
            reduction="mean",
            gridded_operations=LatLonOperations(area_weights_host),
        )

        # Compute loss
        result_serial = loss_fn(x_host, y_host)

        # Aggregate and get logs
        aggregator = MeanAggregator(LatLonOperations(area_weights_host))
        aggregator.record_batch(
            loss=result_serial,
            target_data=example_data_host,
            gen_data=example_data_host,
            target_data_norm=example_data_host,
            gen_data_norm=example_data_host,
        )
        logs_serial = aggregator.get_logs(label="metrics")
        loss_serial = logs_serial["metrics/loss"]

    rel_diff = np.abs(loss_serial - loss_parallel) / loss_serial
    assert rel_diff < error_tol, (
        f"Loss computation mismatch between serial and parallel execution. "
        f"Serial loss: {loss_serial}, Parallel loss: {loss_parallel}, "
        f"Relative difference: {rel_diff}, Tolerance: {error_tol}"
    )
