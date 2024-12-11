import datetime
import logging
from typing import Callable, Dict, MutableMapping, Optional

import torch

from fme.core import metrics
from fme.core.climate_data import ClimateData
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.device import get_device

DerivedVariableFunc = Callable[
    [ClimateData, HybridSigmaPressureCoordinate, datetime.timedelta], torch.Tensor
]


_DERIVED_VARIABLE_REGISTRY: MutableMapping[str, DerivedVariableFunc] = {}


def register(func: DerivedVariableFunc):
    label = func.__name__
    if label in _DERIVED_VARIABLE_REGISTRY:
        raise ValueError(f"Function {label} has already been added to registry.")
    _DERIVED_VARIABLE_REGISTRY[label] = func
    return func


@register
def surface_pressure_due_to_dry_air(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return metrics.surface_pressure_due_to_dry_air(
        data.specific_total_water,
        data.surface_pressure,
        vertical_coordinate,
    )


@register
def surface_pressure_due_to_dry_air_absolute_tendency(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    ps_dry = surface_pressure_due_to_dry_air(data, vertical_coordinate, timestep)
    abs_ps_dry_tendency = torch.zeros_like(ps_dry)
    abs_ps_dry_tendency[:, 1:] = torch.diff(ps_dry, n=1, dim=1).abs()
    return abs_ps_dry_tendency


@register
def total_water_path(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return vertical_coordinate.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
    )


@register
def total_water_path_budget_residual(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
):
    total_water_path = vertical_coordinate.vertical_integral(
        data.specific_total_water,
        data.surface_pressure,
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


@register
def net_energy_flux_toa_into_atmosphere(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
):
    return (
        data.toa_down_sw_radiative_flux
        - data.toa_up_sw_radiative_flux
        - data.toa_up_lw_radiative_flux
    )


@register
def net_energy_flux_sfc_into_atmosphere(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
):
    # property is defined as positive into surface, but want to compare to
    # MSE tendency defined as positive into atmosphere
    return -data.net_surface_energy_flux_without_frozen_precip


@register
def net_energy_flux_into_atmospheric_column(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
):
    return net_energy_flux_sfc_into_atmosphere(
        data, vertical_coordinate, timestep
    ) + net_energy_flux_toa_into_atmosphere(data, vertical_coordinate, timestep)


@register
def column_moist_static_energy(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
):
    return vertical_coordinate.vertical_integral(
        data.moist_static_energy(vertical_coordinate),
        data.surface_pressure,
    )


@register
def column_moist_static_energy_tendency(
    data: ClimateData,
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
):
    mse = column_moist_static_energy(data, vertical_coordinate, timestep)
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
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
    label: str,
    derived_variable_func: DerivedVariableFunc,
    forcing_data: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Computes a derived variable and adds it to the given data.

    The derived variable name must not already exist in the data.

    If any required input data are not available,
    the derived variable will not be computed.

    Args:
        data: dictionary of data add the derived variable to.
        vertical_coordinate: the vertical coordinate.
        timestep: Timestep of the model.
        label: the name of the derived variable.
        derived_variable_func: derived variable function to compute.
        forcing_data: optional dictionary of forcing data needed for some derived
            variables. If necessary forcing inputs are missing, the derived
            variable will not be computed.

    Returns:
        A new data dictionary with the derived variable added.
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
        output = derived_variable_func(climate_data, vertical_coordinate, timestep)
    except KeyError as key_error:
        logging.debug(f"Could not compute {label} because {key_error} is missing")
    else:  # if no exception was raised
        new_data[label] = output
    return new_data


def compute_derived_quantities(
    data: Dict[str, torch.Tensor],
    vertical_coordinate: HybridSigmaPressureCoordinate,
    timestep: datetime.timedelta,
    forcing_data: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Computes all derived quantities from the given data."""
    for label, func in _DERIVED_VARIABLE_REGISTRY.items():
        data = _compute_derived_variable(
            data,
            vertical_coordinate,
            timestep,
            label,
            func,
            forcing_data=forcing_data,
        )
    return data
