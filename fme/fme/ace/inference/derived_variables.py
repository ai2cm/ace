import dataclasses
import datetime
import logging
from typing import Callable, Dict, List, MutableMapping, Optional

import torch

from fme.core import metrics
from fme.core.climate_data import ClimateData
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.device import get_device


@dataclasses.dataclass
class DerivedVariableRegistryEntry:
    func: Callable[[ClimateData, SigmaCoordinates, datetime.timedelta], torch.Tensor]
    required_inputs: Optional[List[str]] = None


_DERIVED_VARIABLE_REGISTRY: MutableMapping[str, DerivedVariableRegistryEntry] = {}


def register(
    required_inputs: Optional[List[str]] = None,
):
    """Decorator for registering a function that computes a derived variable.
    Args:
        required_inputs: refers to the keys of CLIMATE_FIELD_NAME_PREFIXES
            in fme.core.climate_data
    """

    def decorator(
        func: Callable[
            [ClimateData, SigmaCoordinates, datetime.timedelta], torch.Tensor
        ]
    ):
        label = func.__name__
        if label in _DERIVED_VARIABLE_REGISTRY:
            raise ValueError(f"Function {label} has already been added to registry.")
        _DERIVED_VARIABLE_REGISTRY[label] = DerivedVariableRegistryEntry(
            func=func, required_inputs=required_inputs
        )
        return func

    return decorator


@register()
def surface_pressure_due_to_dry_air(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return metrics.surface_pressure_due_to_dry_air(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


@register()
def total_water_path(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return metrics.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


@register()
def total_water_path_budget_residual(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
):
    total_water_path = metrics.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )
    twp_total_tendency = (total_water_path[:, 1:] - total_water_path[:, :-1]) / (
        timestep.total_seconds()
    )
    twp_budget_residual = torch.zeros_like(total_water_path)
    # no budget residual on initial step
    twp_budget_residual[:, 1:] = twp_total_tendency - (
        data.evaporation_rate[:, 1:]
        - data.precipitation_rate[:, 1:]
        + data.tendency_of_total_water_path_due_to_advection[:, 1:]
    )
    return twp_budget_residual


@register(
    required_inputs=[
        "toa_down_sw_radiative_flux",
    ]
)
def net_energy_flux_toa_into_atmosphere(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
):
    return (
        data.toa_down_sw_radiative_flux
        - data.toa_up_sw_radiative_flux
        - data.toa_up_lw_radiative_flux
    )


@register()
def net_energy_flux_sfc_into_atmosphere(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
):
    # property is defined as positive into surface, but want to compare to
    # MSE tendency defined as positive into atmosphere
    return -data.net_surface_energy_flux_without_frozen_precip


@register(
    required_inputs=[
        "toa_down_sw_radiative_flux",
    ]
)
def net_energy_flux_into_atmospheric_column(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
):
    return net_energy_flux_sfc_into_atmosphere(
        data, sigma_coordinates, timestep
    ) + net_energy_flux_toa_into_atmosphere(data, sigma_coordinates, timestep)


@register(required_inputs=["surface_height"])
def column_moist_static_energy(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
):
    return metrics.vertical_integral(
        data.moist_static_energy(sigma_coordinates),
        data.surface_pressure,
        sigma_coordinates.ak,
        sigma_coordinates.bk,
    )


@register(required_inputs=["surface_height"])
def column_moist_static_energy_tendency(
    data: ClimateData,
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
):
    mse = column_moist_static_energy(data, sigma_coordinates, timestep)
    diff = torch.diff(mse, n=1, dim=1)
    # Only the very first timestep in series is filled with nan; subsequent batches
    # drop the first step as it's the initial condition.
    first_step_shape = (diff.shape[0], 1, *diff.shape[2:])
    tendency = (
        torch.concat([torch.zeros(first_step_shape).to(get_device()), diff], dim=1)
        / timestep.total_seconds()
    )
    return tendency


def _compute_derived_variable(
    data: Dict[str, torch.Tensor],
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
    label: str,
    derived_variable: DerivedVariableRegistryEntry,
    forcing_data: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Computes a derived variable and adds it to the given data.

    If the required input data is not available,
    no change will be made to the data.

    Args:
        data: dictionary of data add the derived variable to.
        sigma_coordinates: the vertical coordinate.
        timestep: Timestep of the model.
        label: the name of the derived variable.
        derived_variable: class indicating required names and function to compute.
        forcing_data: optional dictionary of forcing data needed for some derived
            variables. If necessary forcing inputs are missing, the derived
            variable will not be computed.

    Returns:
        A new TrainOutput instance with the derived variable added.

    Note:
        Derived variables are only computed for the denormalized data in stepped.
    """
    if label in data:
        raise ValueError(
            f"Variable {label} already exists. It is not permitted "
            "to overwrite existing variables with derived variables."
        )
    new_data = data.copy()
    if forcing_data is not None:
        for key, value in forcing_data.items():
            if key not in data:
                data[key] = value

    climate_data = ClimateData(data)

    try:
        output = derived_variable.func(climate_data, sigma_coordinates, timestep)
    except KeyError as key_error:
        logging.debug(f"Could not compute {label} because {key_error} is missing")
    else:  # if no exception was raised
        new_data[label] = output
    return new_data


def compute_derived_quantities(
    data: Dict[str, torch.Tensor],
    sigma_coordinates: SigmaCoordinates,
    timestep: datetime.timedelta,
    forcing_data: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Computes all derived quantities from the given data."""
    for label, derived_variable in _DERIVED_VARIABLE_REGISTRY.items():
        data = _compute_derived_variable(
            data,
            sigma_coordinates,
            timestep,
            label,
            derived_variable,
            forcing_data=forcing_data,
        )
    return data
