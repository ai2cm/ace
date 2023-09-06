import dataclasses
import logging
from typing import Callable, MutableMapping, Sequence

import torch
from toolz import curry

from fme.core import metrics
from fme.core.aggregator.climate_data import ClimateData
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.stepper import SteppedData


@dataclasses.dataclass
class DerivedVariableRegistryEntry:
    func: Callable[[ClimateData, SigmaCoordinates], torch.Tensor]
    required_names: Sequence[str]


_DERIVED_VARIABLE_REGISTRY: MutableMapping[str, DerivedVariableRegistryEntry] = {}


@curry
def register(
    required_names: Sequence[str],
    func: Callable[[ClimateData, SigmaCoordinates], torch.Tensor],
):
    """Decorator for registering a function that computes a derived variable."""
    label = func.__name__
    if label in _DERIVED_VARIABLE_REGISTRY:
        raise ValueError(f"Function {label} has already been added to registry.")
    _DERIVED_VARIABLE_REGISTRY[label] = DerivedVariableRegistryEntry(
        func=func, required_names=required_names
    )
    return func


@register(["specific_total_water", "surface_pressure"])
def surface_pressure_due_to_dry_air(
    data: ClimateData, sigma_coordinates: SigmaCoordinates
) -> torch.Tensor:
    return metrics.surface_pressure_due_to_dry_air(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


@register(["specific_total_water", "surface_pressure"])
def total_water_path(
    data: ClimateData, sigma_coordinates: SigmaCoordinates
) -> torch.Tensor:
    return metrics.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


def _compute_derived_variable(
    stepped: SteppedData,
    sigma_coordinates: SigmaCoordinates,
    label: str,
    derived_variable: DerivedVariableRegistryEntry,
) -> SteppedData:
    """Computes a derived variable and adds it to the given data.

    Args:
        stepped: SteppedData instance to add the derived variable to.
        sigma_coordinates: the vertical coordinate.
        label: the name of the derived variable.
        derived_variable: class indicating required names and function to compute.

    Returns:
        A new SteppedData instance with the derived variable added.

    Note:
        Derived variables are only computed for the denormalized data in stepped.
    """
    new_stepped = stepped.copy()
    for data_type in ["gen_data", "target_data"]:
        if label in getattr(stepped, data_type):
            raise ValueError(
                f"Variable {label} already exists in {data_type}. It is not permitted "
                "to overwrite existing variables with derived variables."
            )
        climate_data = ClimateData(getattr(stepped, data_type))
        for name in derived_variable.required_names:
            if not hasattr(climate_data, name) or getattr(climate_data, name) is None:
                logging.warning(
                    f"Could not compute {label} because {name} is missing "
                    f"for {data_type}."
                )
                return stepped
        output = derived_variable.func(climate_data, sigma_coordinates)
        getattr(new_stepped, data_type)[label] = output
    return new_stepped


def compute_derived_quantities(
    stepped: SteppedData,
    sigma_coordinates: SigmaCoordinates,
    registry: MutableMapping[
        str, DerivedVariableRegistryEntry
    ] = _DERIVED_VARIABLE_REGISTRY,
) -> SteppedData:
    """Computes all derived quantities from the given data."""
    for label, derived_variable in registry.items():
        stepped = _compute_derived_variable(
            stepped, sigma_coordinates, label, derived_variable
        )
    return stepped
