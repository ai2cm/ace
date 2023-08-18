import pytest
import torch

import fme
from fme.core.metrics import surface_pressure_due_to_dry_air, vertical_integral


def _get_lats(num_lat: int):
    """A simple example of latitudes."""
    return torch.linspace(-89.5, 89.5, num_lat)


test_cases = (
    "variable, time, lats, n_lon",
    [
        (2, 1, _get_lats(2), 1),
        (1, 2, _get_lats(2), 1),
        (1, 2, _get_lats(2), 4),
    ],
)


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
    lats = _get_lats(num_lat)
    result = fme.spherical_area_weights(lats, num_lon)
    assert torch.all(torch.isclose(result, expected))


@pytest.mark.parametrize(*test_cases)
def test_weighted_mean(variable, time, lats, n_lon):
    """Tests the weighted mean for a few simple test cases."""
    x = torch.randn(time, variable, len(lats), n_lon)
    weights = fme.spherical_area_weights(lats, n_lon)

    result = fme.weighted_mean(x, weights, dim=(0, 2, 3))
    assert result.shape == (variable,), "You should be able to specify time as dim = 1."

    result = fme.weighted_mean(
        torch.zeros(variable, time, len(lats), n_lon), weights, dim=(0, 2, 3)
    )
    assert torch.all(
        torch.isclose(result, torch.tensor([0.0]))
    ), "Weighted mean of zeros should be zero."

    result = fme.weighted_mean(torch.ones(variable, time, len(lats), n_lon), weights)
    assert torch.all(
        torch.isclose(result, torch.Tensor([1.0]))
    ), "The weighted mean of a constant should be that constant."


@pytest.mark.parametrize(*test_cases)
def test_weighted_mean_bias(variable, time, lats, n_lon):
    """Tests the weighted mean bias for a few simple test cases."""
    x = torch.randn(time, variable, len(lats), n_lon)
    y = torch.randn(time, variable, len(lats), n_lon)
    weights = fme.spherical_area_weights(lats, n_lon)

    result = fme.weighted_mean_bias(x, x.clone(), weights, dim=(0, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.tensor(0.0))
    ), "Weighted global mean bias between identical tensors should be zero."
    assert result.shape == (variable,), "You should be able to specify time as dim = 1."

    x = torch.zeros(time, variable, len(lats), n_lon)
    y = torch.ones(time, variable, len(lats), n_lon)

    result = fme.weighted_mean_bias(x, y, weights)
    assert torch.all(
        torch.isclose(result, torch.Tensor([1.0]))
    ), "The weighted mean of a constant should be that constant."

    result = fme.weighted_mean_bias(x, y)
    assert result.shape == tuple(), "Should also work if you do not specify weights."

    x = torch.randn(variable, time, n_lon, len(lats))
    y = torch.randn(variable, time, n_lon, len(lats))
    result = fme.weighted_mean_bias(x, x.clone(), weights.t(), dim=(1, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.tensor(0.0))
    ), "Weighted global mean bias between identical tensors should be zero."
    assert result.shape == (
        variable,
    ), "Swapping dims shouldn't change the final shape."


@pytest.mark.parametrize(*test_cases)
def test_root_mean_squared_error(variable, time, lats, n_lon):
    """Tests the mean squared error for a few simple test cases."""
    x = torch.randn(variable, time, len(lats), n_lon)
    random_weights = torch.rand(len(lats), n_lon)

    result = fme.root_mean_squared_error(x, x.clone(), dim=(0, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.tensor(0.0))
    ), "Root mean squared error between identical tensors should be zero."

    result = fme.root_mean_squared_error(
        torch.zeros(variable, time, len(lats), n_lon),
        torch.ones(variable, time, len(lats), n_lon),
    )
    assert torch.all(
        torch.isclose(result, torch.tensor(1.0))
    ), "Root mean squared error between zero and one should be one."

    result = fme.root_mean_squared_error(
        torch.zeros(variable, time, len(lats), n_lon),
        torch.ones(variable, time, len(lats), n_lon),
        weights=random_weights,
    )
    assert torch.isclose(
        result, torch.tensor([1.0]).sqrt()
    ), "Root mean squared error between zero and one should be one."


@pytest.mark.parametrize(*test_cases)
def test_rmse_of_time_mean(variable, time, lats, n_lon):
    x = torch.randn(variable, time, len(lats), n_lon)
    random_weights = torch.rand(len(lats), n_lon)

    result = fme.rmse_of_time_mean(x, x.clone(), time_dim=1)
    torch.testing.assert_close(
        result,
        torch.zeros((variable,)),
        msg="RMSE of time mean between identical tensors should be zero.",
    )

    result = fme.rmse_of_time_mean(
        torch.zeros(variable, time, len(lats), n_lon),
        torch.ones(variable, time, len(lats), n_lon),
        weights=random_weights,
        time_dim=1,
    )
    torch.testing.assert_close(
        result,
        torch.ones((variable,)),
        msg="RMSE of time mean between zero and 1 should be 1.",
    )


@pytest.mark.parametrize(*test_cases)
def test_time_and_global_mean(variable, time, lats, n_lon):
    x = torch.randn(variable, time, len(lats), n_lon)
    random_weights = torch.rand(len(lats), n_lon)

    result = fme.time_and_global_mean_bias(x, x.clone(), time_dim=1)
    torch.testing.assert_close(
        result,
        torch.zeros((variable,)),
        msg="Time and global mean bias between identical tensors should be zero.",
    )

    result = fme.time_and_global_mean_bias(
        torch.zeros(variable, time, len(lats), n_lon),
        torch.ones(variable, time, len(lats), n_lon),
        weights=random_weights,
        time_dim=1,
    )
    torch.testing.assert_close(
        result,
        torch.ones((variable,)),
        msg="Global mean bias between zero and 1 should be 1.",
    )


def test_gradient_magnitude():
    constant = torch.ones((5, 2, 4, 4))
    constant_grad_magnitude = fme.gradient_magnitude(constant, dim=(-2, -1))
    torch.testing.assert_close(constant_grad_magnitude, torch.zeros_like(constant))

    monotonic = torch.tile(torch.arange(4.0), (5, 2, 4, 1))
    monotonic_grad_magnitude = fme.gradient_magnitude(monotonic, dim=(-2, -1))
    torch.testing.assert_close(monotonic_grad_magnitude, torch.ones_like(monotonic))


def test_weighted_mean_gradient_magnitude():
    constant = torch.ones((5, 2, 4, 4))
    constant_grad_magnitude = fme.weighted_mean_gradient_magnitude(
        constant, dim=(-2, -1)
    )
    torch.testing.assert_close(constant_grad_magnitude, torch.zeros((5, 2)))

    monotonic = torch.tile(torch.arange(4.0), (5, 2, 4, 1))
    monotonic_grad_magnitude = fme.weighted_mean_gradient_magnitude(
        monotonic, dim=(-2, -1)
    )
    torch.testing.assert_close(monotonic_grad_magnitude, torch.ones((5, 2)))


def test_gradient_magnitude_percent_diff():
    constant = torch.ones((5, 2, 4, 4))
    self_percent_diff = fme.gradient_magnitude_percent_diff(
        constant, constant, dim=(-2, -1)
    )
    assert torch.all(torch.isnan(self_percent_diff))

    monotonic = torch.tile(torch.arange(4.0), (5, 2, 4, 1))
    percent_diff = fme.gradient_magnitude_percent_diff(
        monotonic, constant, dim=(-2, -1)
    )
    torch.testing.assert_close(percent_diff, -100 * torch.ones((5, 2)))


def test_vertical_integral_shape():
    nlat, nlon, nz = 4, 8, 3
    water = torch.rand(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = torch.arange(nz + 1), torch.arange(nz + 1)
    water_path = vertical_integral(water, pressure, ak, bk)
    assert water_path.shape == (nlat, nlon)


def test_dry_air_shapes():
    nlat, nlon, nz = 4, 8, 3
    water = torch.rand(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = torch.arange(nz + 1), torch.arange(nz + 1)

    dry_air = surface_pressure_due_to_dry_air(water, pressure, ak, bk)
    assert dry_air.shape == (nlat, nlon)
