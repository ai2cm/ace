import dataclasses
import logging
from typing import Callable, MutableMapping

import torch
from toolz import curry

from fme.core import metrics
from fme.core.climate_data import ClimateData
from fme.core.constants import TIMESTEP_SECONDS
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.stepper import SteppedData


@dataclasses.dataclass
class DerivedVariableRegistryEntry:
    func: Callable[[ClimateData, SigmaCoordinates], torch.Tensor]


_DERIVED_VARIABLE_REGISTRY: MutableMapping[str, DerivedVariableRegistryEntry] = {}


@curry
def register(
    func: Callable[[ClimateData, SigmaCoordinates], torch.Tensor],
):
    """Decorator for registering a function that computes a derived variable."""
    label = func.__name__
    if label in _DERIVED_VARIABLE_REGISTRY:
        raise ValueError(f"Function {label} has already been added to registry.")
    _DERIVED_VARIABLE_REGISTRY[label] = DerivedVariableRegistryEntry(func=func)
    return func


@register()
def surface_pressure_due_to_dry_air(
    data: ClimateData, sigma_coordinates: SigmaCoordinates
) -> torch.Tensor:
    return metrics.surface_pressure_due_to_dry_air(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


@register()
def total_water_path(
    data: ClimateData, sigma_coordinates: SigmaCoordinates
) -> torch.Tensor:
    return metrics.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


@register()
def total_water_path_budget_residual(
    data: ClimateData, sigma_coordinates: SigmaCoordinates
):
    total_water_path = metrics.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )
    twp_total_tendency = (total_water_path[:, 1:] - total_water_path[:, :-1]) / (
        TIMESTEP_SECONDS
    )
    twp_budget_residual = torch.zeros_like(total_water_path)
    # no budget residual on initial step
    twp_budget_residual[:, 1:] = twp_total_tendency - (
        data.evaporation_rate[:, 1:]
        - data.precipitation_rate[:, 1:]
        + data.tendency_of_total_water_path_due_to_advection[:, 1:]
    )
    return twp_budget_residual


def _compute_derived_variable(
    stepped: SteppedData,
    sigma_coordinates: SigmaCoordinates,
    label: str,
    derived_variable: DerivedVariableRegistryEntry,
) -> SteppedData:
    """Computes a derived variable and adds it to the given data.

    If the required input data is not available, a warning will be logged and
    no change will be made to the data.

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
        try:
            output = derived_variable.func(climate_data, sigma_coordinates)
        except KeyError as key_error:
            logging.warning(
                f"Could not compute {label} because {key_error} is missing "
                f"for {data_type}."
            )
        else:  # if no exception was raised
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
