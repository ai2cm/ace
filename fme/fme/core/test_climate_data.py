"""Tests for classes and methods related to climate data."""

import pytest
import torch

from fme.core.climate_data import ClimateData, natural_sort


@pytest.mark.parametrize(
    "names, sorted_names",
    [
        (
            ["a_1", "b_1", "c_1", "a_2"],
            [
                "a_1",
                "a_2",
                "b_1",
                "c_1",
            ],
        ),
        (
            [
                "a_0",
                "a_1",
                "a_12",
                "a_2",
            ],
            [
                "a_0",
                "a_1",
                "a_2",
                "a_12",
            ],
        ),
        (
            [
                "a_0001",
                "a_0012",
                "a_0002",
            ],
            [
                "a_0001",
                "a_0002",
                "a_0012",
            ],
        ),
        (
            [
                "ab1",
                "aa10",
                "aa2",
            ],
            ["aa2", "aa10", "ab1"],
        ),
    ],
)
def test_natural_sort(names, sorted_names):
    assert natural_sort(names) == sorted_names


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
