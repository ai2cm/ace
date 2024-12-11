import numpy as np
import pytest
import torch
import torch_harmonics

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.metrics import (
    net_surface_energy_flux,
    quantile,
    spherical_power_spectrum,
    surface_pressure_due_to_dry_air,
)


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


def test_weighted_std_constant_weights():
    """Tests the weighted std with constant weights."""
    torch.manual_seed(0)
    n_lat = 4
    n_lon = 5
    x = torch.randn(2, 3, n_lat, n_lon)
    weights = torch.ones(n_lat, n_lon)
    result = fme.weighted_std(x, weights, dim=(0, 2, 3))
    # must use numpy std here as we want the standard deviation of
    # the sample, not the population
    unweighted_std = np.std(x.cpu().numpy(), axis=(0, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.as_tensor(unweighted_std))
    ), "Weighted std with constant weights should be same as unweighted std."


def test_weighted_std_none_weights():
    """Tests the weighted std without passing weights."""
    torch.manual_seed(0)
    n_lat = 4
    n_lon = 5
    x = torch.randn(2, 3, n_lat, n_lon)
    weights = None
    result = fme.weighted_std(x, weights, dim=(0, 2, 3))
    # must use numpy std here as we want the standard deviation of
    # the sample, not the population
    unweighted_std = np.std(x.cpu().numpy(), axis=(0, 2, 3))
    assert torch.all(
        torch.isclose(result, torch.as_tensor(unweighted_std))
    ), "Weighted std with constant weights should be same as unweighted std."


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


def test_dry_air_shapes():
    nlat, nlon, nz = 4, 8, 3
    water = torch.rand(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = torch.arange(nz + 1), torch.arange(nz + 1)
    coords = HybridSigmaPressureCoordinate(ak, bk)
    dry_air = surface_pressure_due_to_dry_air(water, pressure, coords)
    assert dry_air.shape == (nlat, nlon)


def single_level_ak_bk():
    ak = torch.zeros(size=[2])
    bk = torch.asarray([0.0, 1.0])
    return ak, bk


def test_single_level_dry_air_no_water():
    torch.manual_seed(0)
    nlat, nlon, nz = 4, 8, 1
    water = torch.zeros(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = single_level_ak_bk()
    coords = HybridSigmaPressureCoordinate(ak, bk)
    dry_air = surface_pressure_due_to_dry_air(water, pressure, coords)
    np.testing.assert_allclose(dry_air.cpu().numpy(), pressure.cpu().numpy())


def test_single_level_dry_air_all_water():
    torch.manual_seed(0)
    nlat, nlon, nz = 4, 8, 1
    water = torch.ones(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = single_level_ak_bk()
    coords = HybridSigmaPressureCoordinate(ak, bk)
    dry_air = surface_pressure_due_to_dry_air(water, pressure, coords)
    np.testing.assert_almost_equal(dry_air.cpu().numpy(), 0.0, decimal=6)


def test_single_level_dry_air_some_water():
    torch.manual_seed(0)
    nlat, nlon, nz = 4, 8, 1
    water = torch.rand(nlat, nlon, nz)
    pressure = torch.rand(nlat, nlon)
    ak, bk = single_level_ak_bk()
    target_dry_air = pressure * (1.0 - water[:, :, 0])
    coords = HybridSigmaPressureCoordinate(ak, bk)
    dry_air = surface_pressure_due_to_dry_air(water, pressure, coords)
    np.testing.assert_allclose(
        dry_air.cpu().numpy(), target_dry_air.cpu().numpy(), rtol=1e-5
    )


def test_net_surface_energy_flux():
    sfc_down_lw_radiative_flux = torch.tensor([100.0])
    sfc_up_lw_radiative_flux = torch.tensor([50.0])
    sfc_down_sw_radiative_flux = torch.tensor([25.0])
    sfc_up_sw_radiative_flux = torch.tensor([10.0])
    latent_heat_flux = torch.tensor([5.0])
    sensible_heat_flux = torch.tensor([2.5])
    expected_net_surface_energy_flux = torch.tensor([57.5])
    result = net_surface_energy_flux(
        sfc_down_lw_radiative_flux,
        sfc_up_lw_radiative_flux,
        sfc_down_sw_radiative_flux,
        sfc_up_sw_radiative_flux,
        latent_heat_flux,
        sensible_heat_flux,
    )
    torch.testing.assert_close(result, expected_net_surface_energy_flux)


@pytest.mark.parametrize(
    "bins, hist, pct, expected",
    [
        pytest.param(
            np.array([0, 1]), np.array([10]), 0.8, 0.8, id="single_bin_interpolation"
        ),
        pytest.param(
            np.array([0, 1, 2]),
            np.array([10, 10]),
            0.8,
            1.6,
            id="two_bin_interpolation",
        ),
        pytest.param(
            np.array([0, 1, 2]), np.array([10, 10]), 1.0, 2.0, id="two_bin_end_value"
        ),
    ],
)
def test_quantile(bins, hist, pct, expected):
    result = quantile(bins, hist, pct)
    np.testing.assert_allclose(result, expected)


def test_spherical_power_spectrum():
    nlat, nlon = 4, 8
    data = torch.rand(2, nlat, nlon)
    sht = torch_harmonics.RealSHT(nlat, nlon)
    spectrum = spherical_power_spectrum(data, sht)
    assert spectrum.shape == (2, nlat - 1)
