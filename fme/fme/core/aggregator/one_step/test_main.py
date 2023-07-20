from fme.core.aggregator.one_step import OneStepAggregator
from fme.core.device import get_device

import torch
import pytest


def test_labels_exist():
    n_sample = 10
    n_time = 3
    nx = 2
    ny = 2
    loss = 1.0
    area_weights = torch.ones(ny).to(get_device())
    agg = OneStepAggregator(area_weights)
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_logs(label="test")
    assert "test/mean/loss" in logs
    assert "test/mean/l1/a" in logs
    assert "test/mean/weighted_rmse/a" in logs
    assert "test/mean/weighted_bias/a" in logs
    assert "test/mean/weighted_grad_mag_percent_diff/a" in logs
    assert "test/snapshot/image-full-field/a" in logs
    assert "test/snapshot/image-residual/a" in logs
    assert "test/snapshot/image-error/a" in logs


def test_loss():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    torch.manual_seed(0)
    example_data = {
        "a": torch.randn(1, 2, 5, 5, device=get_device()),
    }
    area_weights = torch.ones(1).to(get_device())
    aggregator = OneStepAggregator(area_weights)
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
    assert logs["metrics/mean/loss"] == 1.5
    aggregator.record_batch(
        loss=3.0,
        target_data=example_data,
        gen_data=example_data,
        target_data_norm=example_data,
        gen_data_norm=example_data,
    )
    logs = aggregator.get_logs(label="metrics")
    assert logs["metrics/mean/loss"] == 2.0


def test_aggregator_raises_on_no_data():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    aggregator = OneStepAggregator([])
    with pytest.raises(ValueError) as excinfo:
        aggregator.record_batch(
            loss=1.0, target_data={}, gen_data={}, target_data_norm={}, gen_data_norm={}
        )
        # check that the raised exception contains the right substring
        assert "No data" in str(excinfo.value)
