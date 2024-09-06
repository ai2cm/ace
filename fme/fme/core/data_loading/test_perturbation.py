import torch

import fme
from fme.core.data_loading.perturbation import (
    ConstantConfig,
    GreensFunctionConfig,
    PerturbationSelector,
)


def test_constant_perturbation_config():
    selector = PerturbationSelector(
        name="constant",
        config={"amplitude": 1.0},
    )
    perturbation = selector.perturbation
    assert isinstance(perturbation, ConstantConfig)
    assert perturbation.amplitude == 1.0
    nx, ny = 5, 5
    lat = torch.arange(nx, device=fme.get_device())
    lon = torch.arange(ny, device=fme.get_device())
    lats, lons = torch.meshgrid(lat, lon)
    ocean_fraction = torch.ones(nx, ny, device=fme.get_device())
    data = torch.ones(nx, ny, device=fme.get_device())
    expected = 2.0 * torch.ones(nx, ny, device=fme.get_device())
    perturbation.apply_perturbation(data, lats, lons, ocean_fraction)
    torch.testing.assert_allclose(data, expected)


def test_green_function_perturbation_config():
    selector = PerturbationSelector(
        name="greens_function",
        config={
            "amplitude": 1.0,
            "lat_center": 0.0,
            "lon_center": 0.0,
            "lat_width": 10.0,
            "lon_width": 10.0,
        },
    )
    perturbation = selector.perturbation
    assert isinstance(perturbation, GreensFunctionConfig)
    assert perturbation.amplitude == 1.0
    assert perturbation.lat_center == 0.0
    assert perturbation.lon_center == 0.0
    assert perturbation.lat_width == 10.0
    assert perturbation.lon_width == 10.0
    nx, ny = 5, 5
    lat = torch.arange(nx, device=fme.get_device())
    lon = torch.arange(ny, device=fme.get_device())
    lats, lons = torch.meshgrid(lat, lon)
    ocean_fraction = torch.ones(nx, ny, device=fme.get_device())
    data = torch.ones(nx, ny, device=fme.get_device())
    perturbation.apply_perturbation(data, lats, lons, ocean_fraction)
