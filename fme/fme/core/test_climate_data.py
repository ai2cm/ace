"""Tests for classes and methods related to climate data."""

import pytest
import torch

from fme.core.climate_data import ClimateData, _height_at_interface, _layer_thickness


def test__layer_thickness():
    pressure_at_interface = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ]
    )
    air_temperature = torch.tensor(
        [
            [[300, 310], [300, 310]],
            [[300, 310], [300, 310]],
        ]
    )
    specific_total_water = torch.full((2, 2, 2), 0.1)
    dz = _layer_thickness(pressure_at_interface, air_temperature, specific_total_water)
    assert dz.shape == (2, 2, 2)
    assert torch.all(dz >= 0.0)


def test__height_at_interface():
    layer_thickness = torch.tensor([[[3, 2], [1, 0.5]], [[3, 2], [1, 0.5]]])
    height_at_surface = torch.tensor([[10, 20], [10, 20]])
    height_at_interface = _height_at_interface(layer_thickness, height_at_surface)
    assert height_at_interface.shape == (2, 2, 3)
    assert torch.equal(
        height_at_interface,
        torch.tensor(
            [[[15, 12, 10], [21.5, 20.5, 20]], [[15, 12, 10], [21.5, 20.5, 20]]]
        ),
    )


@pytest.mark.parametrize("has_water_variable", [True, False])
def test_missing_specific_total_water(has_water_variable):
    """Check shape of specific total water and make sure that it returns None
    when it can't be computed."""
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8

    def _get_data(water: bool):
        if water:
            return {
                "x": torch.rand(n_samples, n_time_steps, nlat, nlon),
                "specific_total_water_0": torch.rand(
                    n_samples, n_time_steps, nlat, nlon
                ),
                "specific_total_water_1": torch.rand(
                    n_samples, n_time_steps, nlat, nlon
                ),
                "PRESsfc": torch.rand(n_samples, n_time_steps, nlat, nlon),
            }
        else:
            return {
                "x": torch.rand(n_samples, n_time_steps, nlat, nlon),
                "PRESsfc": torch.rand(n_samples, n_time_steps, nlat, nlon),
            }

    climate_data = ClimateData(_get_data(water=has_water_variable))

    if has_water_variable:
        assert climate_data.specific_total_water is not None
        assert climate_data.specific_total_water.shape == (
            n_samples,
            n_time_steps,
            nlat,
            nlon,
            2,
        )
    else:
        with pytest.raises(KeyError):
            _ = climate_data.specific_total_water


@pytest.mark.parametrize("missing_water_layer", [True, False])
def test_keyerror_when_missing_specific_total_water_layer(missing_water_layer: bool):
    """Check shape of specific total water and make sure that it returns None
    when it can't be computed."""
    n_samples, n_time_steps, nlat, nlon = 2, 3, 4, 8

    def _get_data(missing_water_layer: bool):
        data = {
            "x": torch.rand(n_samples, n_time_steps, nlat, nlon),
            "specific_total_water_1": torch.rand(n_samples, n_time_steps, nlat, nlon),
            "PRESsfc": torch.rand(n_samples, n_time_steps, nlat, nlon),
        }
        if not missing_water_layer:
            data["specific_total_water_0"] = torch.rand(
                n_samples, n_time_steps, nlat, nlon
            )
        return data

    climate_data = ClimateData(_get_data(missing_water_layer))

    if not missing_water_layer:
        assert climate_data.specific_total_water is not None
        assert climate_data.specific_total_water.shape == (
            n_samples,
            n_time_steps,
            nlat,
            nlon,
            2,
        )
    else:
        with pytest.raises(ValueError):
            _ = climate_data.specific_total_water
