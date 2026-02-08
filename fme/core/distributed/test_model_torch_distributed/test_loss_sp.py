import os

import numpy as np
import pytest
import torch

from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import LossConfig


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
@pytest.mark.parametrize("global_mean_type", [None])
def test_loss_builds_and_runs_with_spatial_parallelism(global_mean_type):
    """Test that loss computation is consistent between
    serial and parallel execution."""
    error_tol = 1e-13
    nx = 8
    ny = 8

    # ========================================================================
    # Phase 1: Serial execution (no spatial parallelism)
    # ========================================================================
    torch.manual_seed(0)
    with Distributed.force_non_distributed():
        # Create test data
        data_tensor = torch.randn(1, 2, nx, ny, device=get_device())
        example_data = {"a": data_tensor}
        area_weights = torch.ones(nx, ny).to(get_device()) * 5

        # Build loss function
        config = LossConfig(global_mean_type=global_mean_type)
        loss_fn = config.build(
            reduction="mean",
            gridded_operations=LatLonOperations(area_weights),
        )

        # Compute loss
        x = torch.randn(1, 2, nx, ny, device=get_device())
        y = torch.randn(1, 2, nx, ny, device=get_device())
        result_serial = loss_fn(x, y)

        # Aggregate and get logs
        aggregator = MeanAggregator(LatLonOperations(area_weights))
        aggregator.record_batch(
            loss=result_serial,
            target_data=example_data,
            gen_data=example_data,
            target_data_norm=example_data,
            gen_data_norm=example_data,
        )
        logs_serial = aggregator.get_logs(label="metrics")
        loss_serial = logs_serial["metrics/loss"]

        # Move tensors to CPU to preserve for parallel phase
        data_tensor_host = data_tensor.cpu()
        x_host = x.cpu()
        y_host = y.cpu()
        area_weights_host = area_weights.cpu()

    # ========================================================================
    # Phase 2: Parallel execution (2-way horizontal spatial parallelism)
    # ========================================================================
    os.environ["H_PARALLEL_SIZE"] = "2"
    os.environ["W_PARALLEL_SIZE"] = "1"

    # Use the same data from serial execution
    area_weights = area_weights_host * 1.0  # Create a copy

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

    # Clean up environment variables
    os.environ.pop("H_PARALLEL_SIZE", None)
    os.environ.pop("W_PARALLEL_SIZE", None)

    # ========================================================================
    # Phase 3: Verify results match
    # ========================================================================
    rel_diff = np.abs(loss_serial - loss_parallel) / loss_serial
    assert rel_diff < error_tol, (
        f"Loss computation mismatch between serial and parallel execution. "
        f"Serial loss: {loss_serial}, Parallel loss: {loss_parallel}, "
        f"Relative difference: {rel_diff}, Tolerance: {error_tol}"
    )
