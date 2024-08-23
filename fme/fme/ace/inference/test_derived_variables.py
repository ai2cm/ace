import datetime

import pytest
import torch

from fme.core.climate_data import ClimateData
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.stepper import SteppedData

from .derived_variables import (
    DerivedVariableRegistryEntry,
    _compute_derived_variable,
    compute_stepped_derived_quantities,
)

TIMESTEP = datetime.timedelta(hours=6)


def test_compute_derived_variable():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    sigma_coordinates = SigmaCoordinates(
        ak=torch.tensor([0.0, 0.0]), bk=torch.tensor([0.0, 1.0])
    )
    derived_variable = DerivedVariableRegistryEntry(
        func=lambda data, *_: data.surface_pressure + data.precipitation_rate
    )
    output_data = _compute_derived_variable(
        fake_data, sigma_coordinates, TIMESTEP, "c", derived_variable
    )
    torch.testing.assert_close(output_data["c"], torch.tensor([3.0]))


def test_compute_derived_variable_raises_value_error_when_overwriting():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    sigma_coordinates = SigmaCoordinates(
        ak=torch.tensor([0.0, 0.0]), bk=torch.tensor([0.0, 1.0])
    )

    def add_surface_pressure_and_precipitation(data: ClimateData, *_) -> torch.Tensor:
        return data.surface_pressure + data.precipitation_rate

    derived_variable = DerivedVariableRegistryEntry(
        func=add_surface_pressure_and_precipitation
    )
    with pytest.raises(ValueError):
        _compute_derived_variable(
            fake_data, sigma_coordinates, TIMESTEP, "PRATEsfc", derived_variable
        )


def test_compute_derived_quantities():
    torch.manual_seed(0)
    fake_data = {
        "PRESsfc": 10.0 + torch.rand(2, 3, 4, 8),
        "specific_total_water_0": torch.rand(2, 3, 4, 8),
        "specific_total_water_1": torch.rand(2, 3, 4, 8),
        "PRATEsfc": torch.rand(2, 3, 4, 8),
        "LHTFLsfc": torch.rand(2, 3, 4, 8),
        "tendency_of_total_water_path_due_to_advection": torch.rand(2, 3, 4, 8),
    }
    data = SteppedData(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=fake_data,
        target_data=fake_data,
        gen_data_norm=fake_data,
        target_data_norm=fake_data,
    )
    sigma_coordinates = SigmaCoordinates(
        ak=torch.tensor([0.0, 0.5, 0.0]),
        bk=torch.tensor([0.0, 0.5, 1.0]),
    )
    out_data = compute_stepped_derived_quantities(data, sigma_coordinates, TIMESTEP)
    for name in (
        "total_water_path_budget_residual",
        "total_water_path",
        "surface_pressure_due_to_dry_air",
    ):
        assert name in out_data.gen_data
        assert name in out_data.target_data
        assert out_data.gen_data[name].shape == (2, 3, 4, 8)
        assert out_data.target_data[name].shape == (2, 3, 4, 8)
