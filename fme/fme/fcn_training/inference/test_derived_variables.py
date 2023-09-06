import pytest
import torch

from fme.core.stepper import SteppedData

from .derived_variables import DerivedVariableRegistryEntry, _compute_derived_variable


def test_compute_derived_variable():
    fake_data = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    data = SteppedData(
        loss=0,
        gen_data=fake_data,
        target_data=fake_data,
        gen_data_norm=fake_data,
        target_data_norm=fake_data,
    )
    sigma_coordinates = None
    derived_variable = DerivedVariableRegistryEntry(
        func=lambda data, _: data.a + data.b, required_names=["a", "b"]
    )
    output_data = _compute_derived_variable(
        data, sigma_coordinates, "c", derived_variable
    )
    torch.testing.assert_close(output_data.gen_data["c"], torch.tensor([3.0]))
    torch.testing.assert_close(output_data.target_data["c"], torch.tensor([3.0]))


def test_compute_derived_variable_raises_value_error_when_overwriting():
    fake_data = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    data = SteppedData(
        loss=0,
        gen_data=fake_data,
        target_data=fake_data,
        gen_data_norm=fake_data,
        target_data_norm=fake_data,
    )
    sigma_coordinates = None
    derived_variable = DerivedVariableRegistryEntry(
        func=lambda data, _: data.a + data.b, required_names=["a", "b"]
    )
    with pytest.raises(ValueError):
        _compute_derived_variable(data, sigma_coordinates, "b", derived_variable)
