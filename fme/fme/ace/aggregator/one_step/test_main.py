import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.one_step import OneStepAggregator
from fme.ace.stepper import TrainOutput
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations


def test_labels_exist():
    n_sample = 10
    n_time = 3
    nx, ny = 2, 2
    loss = 1.0
    area_weights = torch.ones(ny).to(get_device())
    agg = OneStepAggregator(LatLonOperations(area_weights))
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": loss},
            target_data=target_data,
            gen_data=gen_data,
            time=xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"]),
            normalize=lambda x: x,
        ),
    )
    logs = agg.get_logs(label="test")
    assert "test/mean/loss" in logs
    assert "test/mean/weighted_rmse/a" in logs
    assert "test/mean/weighted_bias/a" in logs
    assert "test/mean/weighted_grad_mag_percent_diff/a" in logs
    assert "test/snapshot/image-full-field/a" in logs
    assert "test/snapshot/image-residual/a" in logs
    assert "test/snapshot/image-error/a" in logs


def test_aggregator_raises_on_no_data():
    """
    Basic test the aggregator combines loss correctly
    with multiple batches and no distributed training.
    """
    ny = 2
    area_weights = torch.ones(ny).to(get_device())
    agg = OneStepAggregator(LatLonOperations(area_weights))
    with pytest.raises(ValueError) as excinfo:
        agg.record_batch(
            batch=TrainOutput(
                metrics={"loss": 1.0},
                target_data={},
                gen_data={},
                time=xr.DataArray(np.zeros((0, 0)), dims=["sample", "time"]),
                normalize=lambda x: x,
            ),
        )
        # check that the raised exception contains the right substring
        assert "No data" in str(excinfo.value)


def test__get_loss_scaled_mse_components():
    loss_scaling = {
        "a": torch.tensor(1.0),
        "b": torch.tensor(0.5),
    }
    agg = OneStepAggregator(
        gridded_operations=LatLonOperations(
            area_weights=torch.ones(10).to(get_device())
        ),
        loss_scaling=loss_scaling,
    )

    logs = {
        "test/mean/weighted_rmse/a": 1.0,
        "test/mean/weighted_rmse/b": 4.0,
        "test/mean/weighted_rmse/c": 0.0,
    }
    result = agg._get_loss_scaled_mse_components(logs, "test")
    scaled_squared_errors_sum = (1.0 / 1.0) ** 2 + (4.0 / 0.5) ** 2
    assert (
        result["test/mean/mse_fractional_components/a"] == 1 / scaled_squared_errors_sum
    )
    assert (
        result["test/mean/mse_fractional_components/b"]
        == 64 / scaled_squared_errors_sum
    )
    assert "test/mean/mse_fractional_components/c" not in result
