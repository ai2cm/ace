import numpy as np
import pytest
import torch
import os
from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations

from fme.core.distributed import Distributed


def test_loss_wo_sp():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    nx=8
    ny=8
    torch.manual_seed(0)
    example_data = {
        "a": torch.randn(1, 2, nx, ny, device=get_device()),
    }
    area_weights = torch.ones(nx,ny).to(get_device())
    aggregator = MeanAggregator(LatLonOperations(area_weights))
    aggregator.record_batch(
        loss=1.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    aggregator.record_batch(
        loss=2.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    print("lost", logs["metrics/loss"] )
    assert logs["metrics/loss"] == 1.5
    aggregator.record_batch(
        loss=3.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    print("lost", logs["metrics/loss"] )
    assert logs["metrics/loss"] == 2.0

def test_loss_with_sp():
    os.environ['H_PARALLEL_SIZE'] = '2'
    os.environ['W_PARALLEL_SIZE'] = '2'
    nx=8
    ny=8
    torch.manual_seed(0)
    tensor_data_host=torch.randn(1, 2, nx, ny)
    area_weights = torch.ones(nx,ny)
    aggregator = MeanAggregator(LatLonOperations(area_weights))
    dist = Distributed.get_instance()
    this_shape=(tensor_data_host.shape[-2],tensor_data_host.shape[-1])
    tensor_data_local_host = (tensor_data_host[:,:,*dist.get_local_slices(this_shape)]).detach().clone()
    tensor_data_local=tensor_data_local_host.to(dist.local_rank)

    example_data = {
        "a": tensor_data_local
    }

    aggregator.record_batch(
        loss=1.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    aggregator.record_batch(
        loss=2.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    print("lost", logs["metrics/loss"] )
    assert logs["metrics/loss"] == 1.5
    aggregator.record_batch(
        loss=3.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    print("lost", logs["metrics/loss"] )
    assert logs["metrics/loss"] == 2.0
