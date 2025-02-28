import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.aggregator.one_step import OneStepAggregator
from fme.ace.stepper import TrainOutput
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device


def test_labels_exist():
    n_sample = 10
    n_time = 3
    nx, ny = 2, 2
    loss = 1.0
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    agg = OneStepAggregator(lat_lon_coordinates.to(device=get_device()))
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
    nx, ny = 2, 2
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    agg = OneStepAggregator(lat_lon_coordinates.to(device=get_device()))
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
    nx, ny = 10, 10
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    agg = OneStepAggregator(
        lat_lon_coordinates.to(device=get_device()),
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


@pytest.mark.parametrize(
    "epoch", [pytest.param(None, id="no epoch"), pytest.param(2, id="epoch 2")]
)
def test_flush_diagnostics(tmpdir, epoch):
    nx, ny, n_sample, n_time = 2, 2, 10, 3
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    agg = OneStepAggregator(
        lat_lon_coordinates.to(device=get_device()), output_dir=(tmpdir / "val")
    )
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    time = xr.DataArray(np.zeros((n_sample, n_time)), dims=["sample", "time"])
    agg.record_batch(
        batch=TrainOutput(
            metrics={"loss": 1.0},
            target_data=target_data,
            gen_data=gen_data,
            time=time,
            normalize=lambda x: x,
        ),
    )
    if epoch is not None:
        agg.flush_diagnostics(epoch=epoch)
        output_dir = tmpdir / "val" / f"epoch_{epoch:04d}"
    else:
        agg.flush_diagnostics()
        output_dir = tmpdir / "val"
    expected_files = [
        "mean",
    ]
    for file in expected_files:
        assert (output_dir / f"{file}_diagnostics.nc").exists()
