import pytest
import torch

import metrics

test_cases = (
    "variable, time, lat, lon",
    [
        (2, 1, 2, 1),
        (1, 2, 2, 1),
        (1, 2, 2, 4),
    ],
)


@pytest.mark.parametrize(
    "num_lat_cells, expected",
    [
        (2, torch.tensor([-45.0, 45.0])),
        (4, torch.tensor([-67.5, -22.5, 22.5, 67.5])),
    ],
)
def test_lat_cell_centers(num_lat_cells, expected):
    """Tests the lat cell centers."""
    assert torch.all(torch.isclose(metrics.lat_cell_centers(num_lat_cells), expected))


@pytest.mark.parametrize(
    "num_lat, num_lon, expected",
    [
        (2, 1, torch.tensor([[0.5], [0.5]])),
        (2, 2, torch.tensor([[0.25, 0.25], [0.25, 0.25]])),
        (
            2,
            4,
            torch.tensor(
                [[0.1250, 0.1250, 0.1250, 0.1250], [0.1250, 0.1250, 0.1250, 0.1250]]
            ),
        ),
    ],
)
def test_spherical_area_weights(num_lat, num_lon, expected):
    """Tests the shapes and a couple simple cases of the spherical area weights."""
    result = metrics.spherical_area_weights(num_lat, num_lon)
    assert torch.all(torch.isclose(result, expected))


@pytest.mark.parametrize(*test_cases)
def test_weighted_mean_bias(variable, time, lat, lon):
    """Tests the weighted mean bias for a few simple test cases."""
    x = torch.randn(time, variable, lat, lon)
    y = torch.randn(time, variable, lat, lon)
    weights = metrics.spherical_area_weights(lat, lon)

    result = metrics.weighted_mean_bias(x, x.clone(), weights, dim=(0, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.tensor(0.0))
    ), "Weighted global mean bias between identical tensors should be zero."
    assert result.shape == (variable,), "You should be able to specify time as dim = 1."

    x = torch.zeros(time, variable, lat, lon)
    y = torch.ones(time, variable, lat, lon)

    result = metrics.weighted_mean_bias(x, y, weights)
    assert torch.all(torch.isclose(result, weights.mean())), (
        "Weighted global mean bias between zero and one "
        "should be the mean of the lat weights."
    )

    result = metrics.weighted_mean_bias(x, y)
    assert result.shape == tuple(), "Should also work if you do not specify weights."

    x = torch.randn(variable, time, lon, lat)
    y = torch.randn(variable, time, lon, lat)
    result = metrics.weighted_mean_bias(x, x.clone(), weights.t(), dim=(1, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.tensor(0.0))
    ), "Weighted global mean bias between identical tensors should be zero."
    assert result.shape == (
        variable,
    ), "Swapping dims shouldn't change the final shape."


@pytest.mark.parametrize(*test_cases)
def test_mean_squared_error(variable, time, lat, lon):
    """Tests the mean squared error for a few simple test cases."""
    x = torch.randn(variable, time, lat, lon)
    random_weights = torch.rand(lat, lon)

    result = metrics.mean_squared_error(x, x.clone(), dim=(0, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.tensor(0.0))
    ), "Mean squared error between identical tensors should be zero."

    result = metrics.mean_squared_error(
        torch.zeros(variable, time, lat, lon), torch.ones(variable, time, lat, lon)
    )
    assert torch.all(
        torch.isclose(result, torch.tensor(1.0))
    ), "Mean squared error between zero and one should be one."

    result = metrics.mean_squared_error(
        torch.zeros(variable, time, lat, lon),
        torch.ones(variable, time, lat, lon),
        weights=random_weights,
    )
    assert torch.all(
        torch.isclose(result, random_weights.mean().sqrt())
    ), "Mean squared error between zero and one should be the mean of the weights."
