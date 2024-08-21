import pytest
import torch

from fme.core.aggregator.one_step import OneStepAggregator
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device


def test_labels_exist():
    n_sample = 10
    n_time = 3
    nx, ny, nz = 2, 2, 3
    loss = 1.0
    area_weights = torch.ones(ny).to(get_device())
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    agg = OneStepAggregator(area_weights, sigma_coordinates)
    target_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    target_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    gen_data_norm = {"a": torch.randn(n_sample, n_time, nx, ny, device=get_device())}
    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)
    logs = agg.get_logs(label="test")
    assert "test/mean/loss" in logs
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
    nz = 3
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    aggregator = OneStepAggregator(area_weights, sigma_coordinates)
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
    ny, nz = 2, 3
    area_weights = torch.ones(ny).to(get_device())
    sigma_coordinates = SigmaCoordinates(torch.arange(nz + 1), torch.arange(nz + 1))
    agg = OneStepAggregator(area_weights, sigma_coordinates)
    with pytest.raises(ValueError) as excinfo:
        agg.record_batch(
            loss=1.0, target_data={}, gen_data={}, target_data_norm={}, gen_data_norm={}
        )
        # check that the raised exception contains the right substring
        assert "No data" in str(excinfo.value)


def test_derived(very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    n_sample = 5
    n_time = 3
    nx, ny, nz = 2, 4, 3
    loss = 1.0
    area_weights = torch.ones(ny).to(get_device())
    sigma_coordinates = SigmaCoordinates(
        torch.arange(nz + 1).to(get_device()), torch.arange(nz + 1).to(get_device())
    )
    agg = OneStepAggregator(area_weights, sigma_coordinates)

    def _make_data():
        fields = ["a", "PRESsfc"] + [f"specific_total_water_{i}" for i in range(nz)]
        return {
            field: torch.randn(n_sample, n_time, nx, ny, device=get_device())
            for field in fields
        }

    target_data = _make_data()
    gen_data = _make_data()
    target_data_norm = _make_data()
    gen_data_norm = _make_data()

    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)

    logs = agg.get_logs("")
    target = logs["/derived/surface_pressure_due_to_dry_air/target"]
    gen = logs["/derived/surface_pressure_due_to_dry_air/gen"]

    assert target.shape == ()
    assert not torch.isnan(target).any()
    assert gen.shape == ()
    assert not torch.isnan(target).any()


def test_derived_missing_surface_pressure():
    n_sample = 5
    n_time = 3
    nx, ny, nz = 2, 4, 3
    loss = 1.0
    area_weights = torch.ones(ny).to(get_device())
    sigma_coordinates = SigmaCoordinates(
        torch.arange(nz + 1).to(get_device()), torch.arange(nz + 1).to(get_device())
    )
    agg = OneStepAggregator(area_weights, sigma_coordinates)

    def _make_data():
        fields = ["a"]  # N.B. no surface pressure or water fields.
        return {
            field: torch.randn(n_sample, n_time, nx, ny, device=get_device())
            for field in fields
        }

    target_data = _make_data()
    gen_data = _make_data()
    target_data_norm = _make_data()
    gen_data_norm = _make_data()

    agg.record_batch(loss, target_data, gen_data, target_data_norm, gen_data_norm)

    logs = agg.get_logs("")
    target = logs["/derived/surface_pressure_due_to_dry_air/target"]
    gen = logs["/derived/surface_pressure_due_to_dry_air/gen"]

    assert target.shape == () and torch.isnan(target).all()
    assert gen.shape == () and torch.isnan(target).all()


def test__get_loss_scaled_mse_components():
    x = torch.ones(10).to(get_device())
    loss_scaling = {
        "a": torch.tensor(1.0),
        "b": torch.tensor(0.5),
    }
    agg = OneStepAggregator(
        area_weights=torch.ones(10).to(get_device()),
        sigma_coordinates=SigmaCoordinates(x, x),
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
