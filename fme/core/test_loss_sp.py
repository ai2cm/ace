import pytest
import torch
import os
import numpy as np
from fme.core import metrics
from fme.core.device import get_device
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import (
    AreaWeightedMSELoss,
    CRPSLoss,
    EnergyScoreLoss,
    GlobalMeanLoss,
    LossConfig,
    StepLossConfig,
    VariableWeightingLoss,
    WeightedMappingLoss,
    _construct_weight_tensor,
)
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.ace.aggregator.one_step.reduced import MeanAggregator
from fme.core.distributed import Distributed

@pytest.mark.parametrize("global_mean_type", [None])
def test_loss_builds_and_runs_wo_sp(global_mean_type):
    nx=8
    ny=8
    torch.manual_seed(0)
    data_tensor=torch.randn(1, 2, nx, ny, device=get_device())
    example_data = {
        "a":data_tensor ,
    }
    area_weights = torch.ones(nx,ny).to(get_device())*5

    config = LossConfig(global_mean_type=global_mean_type)
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area_weights),
    )

    x = torch.randn(1, 2, nx, ny, device=get_device())
    y = torch.randn(1, 2, nx, ny, device=get_device())

    result = loss(x, y)

    aggregator = MeanAggregator(LatLonOperations(area_weights))
    aggregator.record_batch(
        loss=result,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    tmp_path="testdata-loss"
    torch.save(area_weights, os.path.join(tmp_path, "area_weights.pt"))
    torch.save(data_tensor, os.path.join(tmp_path, "example_data.pt"))
    torch.save(x, os.path.join(tmp_path, "x.pt"))
    torch.save(y, os.path.join(tmp_path, "y.pt"))
    print("loss", logs["metrics/loss"] )
    torch.save(logs["metrics/loss"], os.path.join(tmp_path, "loss.pt"))

@pytest.mark.parametrize("global_mean_type", [None])
def test_loss_builds_and_runs_with_sp(global_mean_type):
    os.environ['H_PARALLEL_SIZE'] = '2'
    os.environ['W_PARALLEL_SIZE'] = '2'
    nx=8
    ny=8
    tmp_path="testdata-loss"
    tensor_data_host = torch.load(os.path.join(tmp_path, "example_data.pt"))
    x_host=torch.load(os.path.join(tmp_path, "x.pt"))
    y_host=torch.load(os.path.join(tmp_path, "y.pt"))
    loss_serial=torch.load(os.path.join(tmp_path, "loss.pt"))

    torch.manual_seed(0)

    # tensor_data_host=torch.randn(1, 2, nx, ny)
    area_weights = torch.ones(nx,ny)*5.0
    aggregator = MeanAggregator(LatLonOperations(area_weights))
    dist = Distributed.get_instance()
    this_shape=(tensor_data_host.shape[-2],tensor_data_host.shape[-1])
    tensor_data_local_host = (tensor_data_host[:,:,*dist.get_local_slices(this_shape)]).detach().clone()
    tensor_data_local=tensor_data_local_host.to(dist.local_rank)
    example_data = {
        "a": tensor_data_local
    }

    config = LossConfig(global_mean_type=global_mean_type)
    loss = config.build(
        reduction="mean",
        gridded_operations=LatLonOperations(area_weights),
    )

    # x_host = torch.randn(1, 2, nx, ny)
    # y_host = torch.randn(1, 2, nx, ny)

    this_shape_x=(x_host.shape[-2],x_host.shape[-1])
    x_local_host = (x_host[:,:,*dist.get_local_slices(this_shape_x)]).detach().clone()
    x_local=x_local_host.to(dist.local_rank)
    y_local_host = (y_host[:,:,*dist.get_local_slices(this_shape_x)]).detach().clone()
    y_local=y_local_host.to(dist.local_rank)

    result_local = loss(x_local, y_local)

    aggregator = MeanAggregator(LatLonOperations(area_weights))
    aggregator.record_batch(
        loss=result_local,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )

    error_tol=1e-13
    logs = aggregator.get_logs(label="metrics")
    # print("lost", logs["metrics/loss"] )
    # print("loss_serial", loss_serial )
    rel_diff = np.abs(loss_serial -  logs["metrics/loss"])/loss_serial
    assert rel_diff < error_tol
