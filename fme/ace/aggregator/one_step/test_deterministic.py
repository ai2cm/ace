import torch

from fme.ace.aggregator.one_step.deterministic import OneStepDeterministicAggregator
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo


def test__get_loss_scaled_mse_components():
    loss_scaling = {
        "a": torch.tensor(1.0),
        "b": torch.tensor(0.5),
    }
    nx, ny = 10, 10
    lat_lon_coordinates = LatLonCoordinates(torch.arange(nx), torch.arange(ny))
    # keep area weights ones for simplicity
    lat_lon_coordinates._area_weights = torch.ones(nx, ny)
    ds_info = DatasetInfo(horizontal_coordinates=lat_lon_coordinates)
    agg = OneStepDeterministicAggregator(
        ds_info,
        loss_scaling=loss_scaling,
        save_diagnostics=False,
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
