from typing import Any

import pytest
import torch

from fme.core.gridded_ops import GriddedOperations, HEALPixOperations, LatLonOperations
from fme.core.typing_ import TensorMapping


@pytest.mark.parametrize(
    "state, expected_class",
    [
        (
            {
                "type": "LatLonOperations",
                "state": {"area_weights": torch.tensor([[1.0, 1.0], [1.0, 1.0]])},
            },
            LatLonOperations,
        ),
        (
            {
                "type": "HEALPixOperations",
                "state": {},
            },
            HEALPixOperations,
        ),
    ],
)
def test_gridded_operations_from_state(
    state: dict[str, Any],
    expected_class: type[GriddedOperations],
):
    ops = GriddedOperations.from_state(state)
    assert isinstance(ops, expected_class)

    recovered_state = ops.to_state()
    assert recovered_state == state

    with pytest.raises(RuntimeError):
        expected_class.from_state(state["state"])


def test_latlon_area_weighted_sum_dict():
    area_weights = torch.ones(4, 5)
    ops = LatLonOperations(area_weights=area_weights)
    data_dict: TensorMapping = {
        "var1": torch.randn(2, 4, 5),
        "var2": torch.randn(2, 4, 5),
    }
    result = ops.area_weighted_sum_dict(data_dict)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"var1", "var2"}
    for var_name in data_dict:
        assert result[var_name].shape == (2,)
        # all weights are 1, so it's just sum
        torch.testing.assert_close(
            result[var_name], data_dict[var_name].sum(dim=(-2, -1))
        )


def test_latlon_zonal_mean():
    area_weights = torch.ones(4, 5)
    ops = LatLonOperations(area_weights=area_weights)
    n_sample, n_time, n_lat, n_lon = 2, 4, 5, 6
    data = torch.randn(n_sample, n_time, n_lat, n_lon)
    result = ops.zonal_mean(data)
    assert result.shape == (n_sample, n_time, n_lat)
    torch.testing.assert_close(result, data.mean(dim=-1))


def test_latlon_area_weighted_mean_dict():
    area_weights = torch.ones(4, 5)
    ops = LatLonOperations(area_weights=area_weights)
    data_dict: TensorMapping = {
        "var1": torch.randn(2, 4, 5),
        "var2": torch.randn(2, 4, 5),
    }
    result = ops.area_weighted_mean_dict(data_dict)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"var1", "var2"}
    for var_name in data_dict:
        assert result[var_name].shape == (2,)
        # all weights are 1, so it's just mean
        torch.testing.assert_close(
            result[var_name], data_dict[var_name].mean(dim=(-2, -1))
        )


def test_latlon_area_weighted_rmse_dict():
    area_weights = torch.rand(4).unsqueeze(-1).broadcast_to(4, 5)
    ops = LatLonOperations(area_weights=area_weights)
    truth_dict: TensorMapping = {
        "var1": torch.randn(2, 4, 5),
        "var2": torch.randn(2, 4, 5),
    }
    predicted_dict: TensorMapping = {
        "var1": torch.randn(2, 4, 5),
        "var2": torch.randn(2, 4, 5),
    }
    result = ops.area_weighted_rmse_dict(truth_dict, predicted_dict)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"var1", "var2"}

    for var_name in truth_dict:
        assert result[var_name].shape == (2,)
        diff_sq = (predicted_dict[var_name] - truth_dict[var_name]) ** 2
        weighted_diff_sq = diff_sq * area_weights
        mean_weighted_diff_sq = weighted_diff_sq.sum(dim=(-2, -1)) / area_weights.sum()
        expected_rmse_var1 = torch.sqrt(mean_weighted_diff_sq)
        torch.testing.assert_close(result[var_name], expected_rmse_var1)


def test_latlon_area_weighted_std_dict_input():
    area_weights = torch.rand(4).unsqueeze(-1).broadcast_to(4, 5)
    ops = LatLonOperations(area_weights=area_weights)
    data_dict: TensorMapping = {
        "var1": torch.randn(2, 4, 5),
        "var2": torch.randn(2, 4, 5),
    }
    result = ops.area_weighted_std_dict(data_dict)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"var1", "var2"}

    for var_name in data_dict:
        assert result[var_name].shape == (2,)
        mean_var1 = ops.area_weighted_mean(data_dict[var_name], keepdim=True)
        variance = ops.area_weighted_mean((data_dict[var_name] - mean_var1) ** 2)
        expected_std_var1 = variance.sqrt()
        torch.testing.assert_close(result[var_name], expected_std_var1)
