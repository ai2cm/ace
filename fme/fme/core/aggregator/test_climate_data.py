"""Tests for classes and methods related to climate data."""

import pytest
import torch

from fme.core.aggregator.climate_data import ClimateData


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
        assert climate_data.specific_total_water is None


def test_climate_data_attr():
    climate_data = ClimateData(
        {"x": torch.tensor([1.0]), "ps": torch.tensor([2.0])},
        climate_field_name_prefixes={"surface_pressure": "ps"},
    )
    assert climate_data.x == torch.tensor([1.0])
    assert climate_data.ps == torch.tensor([2.0])
    assert climate_data.surface_pressure == torch.tensor([2.0])


def test_climate_data_raises_attribute_error():
    climate_data = ClimateData(
        {"x": torch.tensor([1.0])}, climate_field_name_prefixes={}
    )
    with pytest.raises(AttributeError):
        climate_data.y
