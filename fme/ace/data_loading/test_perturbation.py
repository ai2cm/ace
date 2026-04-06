import torch

import fme
from fme.ace.data_loading.perturbation import (
    ConstantConfig,
    GreensFunctionConfig,
    PerturbationSelector,
)


def test_constant_perturbation_config():
    selector = PerturbationSelector(
        type="constant",
        config={"amplitude": 1.0},
    )
    perturbation = selector.build()
    assert isinstance(perturbation, ConstantConfig)
    assert perturbation.amplitude == 1.0
    nx, ny = 5, 5
    lat = torch.arange(nx, device=fme.get_device())
    lon = torch.arange(ny, device=fme.get_device())
    lats, lons = torch.meshgrid(lat, lon, indexing="ij")
    ocean_fraction = torch.ones(nx, ny, device=fme.get_device())
    data = torch.ones(nx, ny, device=fme.get_device())
    expected = 2.0 * torch.ones(nx, ny, device=fme.get_device())
    perturbation.apply_perturbation(data, lats, lons, ocean_fraction)
    torch.testing.assert_close(data, expected)


def test_constant_perturbation_all_grid_points():
    """With ocean_fraction_cutoff=None, perturbation applies everywhere."""
    selector = PerturbationSelector(
        type="constant",
        config={"amplitude": 2.0, "ocean_fraction_cutoff": None},
    )
    perturbation = selector.build()
    assert isinstance(perturbation, ConstantConfig)
    nx, ny = 5, 5
    lat = torch.arange(nx, device=fme.get_device())
    lon = torch.arange(ny, device=fme.get_device())
    lats, lons = torch.meshgrid(lat, lon, indexing="ij")
    ocean_fraction = torch.zeros(nx, ny, device=fme.get_device())
    data = torch.ones(nx, ny, device=fme.get_device())
    perturbation.apply_perturbation(data, lats, lons, ocean_fraction)
    expected = 3.0 * torch.ones(nx, ny, device=fme.get_device())
    torch.testing.assert_close(data, expected)


def test_constant_perturbation_ocean_only():
    """Default ocean_fraction_cutoff=0.5 skips land points."""
    config = ConstantConfig(amplitude=1.0)
    nx, ny = 4, 4
    lat = torch.arange(nx, device=fme.get_device()).float()
    lon = torch.arange(ny, device=fme.get_device()).float()
    lats, lons = torch.meshgrid(lat, lon, indexing="ij")
    ocean_fraction = torch.zeros(nx, ny, device=fme.get_device())
    ocean_fraction[:2, :] = 1.0  # top half is ocean
    data = torch.zeros(nx, ny, device=fme.get_device())
    config.apply_perturbation(data, lats, lons, ocean_fraction)
    assert torch.all(data[:2, :] == 1.0)
    assert torch.all(data[2:, :] == 0.0)


def test_green_function_perturbation_config():
    selector = PerturbationSelector(
        type="greens_function",
        config={
            "amplitude": 1.0,
            "lat_center": 0.0,
            "lon_center": 0.0,
            "lat_width": 10.0,
            "lon_width": 10.0,
        },
    )
    perturbation = selector.build()
    assert isinstance(perturbation, GreensFunctionConfig)
    assert perturbation.amplitude == 1.0
    assert perturbation.lat_center == 0.0
    assert perturbation.lon_center == 0.0
    assert perturbation.lat_width == 10.0
    assert perturbation.lon_width == 10.0
    nx, ny = 5, 5
    lat = torch.arange(nx, device=fme.get_device())
    lon = torch.arange(ny, device=fme.get_device())
    lats, lons = torch.meshgrid(lat, lon, indexing="ij")
    ocean_fraction = torch.ones(nx, ny, device=fme.get_device())
    data = torch.ones(nx, ny, device=fme.get_device())
    perturbation.apply_perturbation(data, lats, lons, ocean_fraction)
