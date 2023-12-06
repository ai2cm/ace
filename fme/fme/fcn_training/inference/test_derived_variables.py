import pytest
import torch

from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.stepper import SteppedData

from .derived_variables import (
    DerivedVariableRegistryEntry,
    _compute_derived_variable,
    compute_derived_quantities,
)


def test_compute_derived_variable():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    data = SteppedData(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=fake_data,
        target_data=fake_data,
        gen_data_norm=fake_data,
        target_data_norm=fake_data,
    )
    sigma_coordinates = None
    derived_variable = DerivedVariableRegistryEntry(
        func=lambda data, _: data.surface_pressure + data.precipitation_rate
    )
    output_data = _compute_derived_variable(
        data, sigma_coordinates, "c", derived_variable
    )
    torch.testing.assert_close(output_data.gen_data["c"], torch.tensor([3.0]))
    torch.testing.assert_close(output_data.target_data["c"], torch.tensor([3.0]))


def test_compute_derived_variable_raises_value_error_when_overwriting():
    fake_data = {"PRESsfc": torch.tensor([1.0]), "PRATEsfc": torch.tensor([2.0])}
    data = SteppedData(
        metrics={"loss": torch.tensor(0.0)},
        gen_data=fake_data,
        target_data=fake_data,
        gen_data_norm=fake_data,
        target_data_norm=fake_data,
    )
    sigma_coordinates = None
    derived_variable = DerivedVariableRegistryEntry(
        func=lambda data, _: data.surface_pressure + data.precipitation_rate
    )
    with pytest.raises(ValueError):
        _compute_derived_variable(data, sigma_coordinates, "PRATEsfc", derived_variable)


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
    out_data = compute_derived_quantities(data, sigma_coordinates)
    for name in (
        "total_water_path_budget_residual",
        "total_water_path",
        "surface_pressure_due_to_dry_air",
    ):
        assert name in out_data.gen_data
        assert name in out_data.target_data
        assert out_data.gen_data[name].shape == (2, 3, 4, 8)
        assert out_data.target_data[name].shape == (2, 3, 4, 8)
