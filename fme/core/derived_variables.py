import datetime
import logging
from collections.abc import Callable, MutableMapping

import torch

from fme.core.atmosphere_data import AtmosphereData, HasAtmosphereVerticalIntegral
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.ocean_derived_variables import get_ocean_derived_variable_metadata
from fme.core.typing_ import TensorDict

DerivedVariableFunc = Callable[[AtmosphereData, datetime.timedelta], torch.Tensor]


_DERIVED_VARIABLE_REGISTRY: MutableMapping[
    str, tuple[DerivedVariableFunc, VariableMetadata]
] = {}


def get_derived_variable_metadata() -> dict[str, VariableMetadata]:
    return {
        **get_atmosphere_derived_variable_metadata(),
        **get_ocean_derived_variable_metadata(),
    }


def get_atmosphere_derived_variable_metadata() -> dict[str, VariableMetadata]:
    return {
        label: metadata for label, (_, metadata) in _DERIVED_VARIABLE_REGISTRY.items()
    }


def register(metadata: VariableMetadata):
    def decorator(func: DerivedVariableFunc):
        label = func.__name__
        if label in _DERIVED_VARIABLE_REGISTRY:
            raise ValueError(f"Function {label} has already been added to registry.")
        _DERIVED_VARIABLE_REGISTRY[label] = (func, metadata)
        return func

    return decorator


@register(VariableMetadata("Pa", "Surface pressure due to dry air only"))
def surface_pressure_due_to_dry_air(
    data: AtmosphereData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.surface_pressure_due_to_dry_air


@register(
    VariableMetadata("Pa/s", "Absolute value of tendency of dry air surface pressure")
)
def surface_pressure_due_to_dry_air_absolute_tendency(
    data: AtmosphereData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    ps_dry = data.surface_pressure_due_to_dry_air
    abs_ps_dry_tendency = torch.zeros_like(ps_dry)
    abs_ps_dry_tendency[:, 1:] = torch.diff(ps_dry, n=1, dim=1).abs()
    return abs_ps_dry_tendency


@register(VariableMetadata("kg/m**2", "Total water path"))
def total_water_path(
    data: AtmosphereData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.total_water_path


@register(VariableMetadata("kg/m**2/s", "Total water path budget residual"))
def total_water_path_budget_residual(
    data: AtmosphereData,
    timestep: datetime.timedelta,
):
    total_water_path = data.total_water_path
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


@register(VariableMetadata("W/m**2", "Net TOA radiative flux into atmosphere"))
def net_energy_flux_toa_into_atmosphere(
    data: AtmosphereData,
    timestep: datetime.timedelta,
):
    return data.net_top_of_atmosphere_energy_flux


@register(VariableMetadata("W/m**2", "Net surface energy flux into atmosphere"))
def net_energy_flux_sfc_into_atmosphere(
    data: AtmosphereData,
    timestep: datetime.timedelta,
):
    # property is defined as positive into surface, but want to compare to
    # MSE tendency defined as positive into atmosphere
    return -data.net_surface_energy_flux


@register(
    VariableMetadata(
        "W/m**2", "Net energy flux through TOA and surface into atmosphere"
    )
)
def net_energy_flux_into_atmospheric_column(
    data: AtmosphereData, timestep: datetime.timedelta
):
    return data.net_energy_flux_into_atmosphere


@register(VariableMetadata("J/m**2", "Total energy path following ACE2 assumptions"))
def total_energy_ace2_path(
    data: AtmosphereData,
    timestep: datetime.timedelta,
):
    return data.total_energy_ace2_path


@register(
    VariableMetadata(
        "W/m**2", "Tendency of total energy path following ACE2 assumptions"
    )
)
def total_energy_ace2_path_tendency(
    data: AtmosphereData,
    timestep: datetime.timedelta,
):
    mse = total_energy_ace2_path(data, timestep)
    mse_tendency = torch.zeros_like(mse)
    mse_tendency[:, 1:] = torch.diff(mse, n=1, dim=1) / timestep.total_seconds()
    return mse_tendency


@register(
    VariableMetadata(
        "W/m**2",
        "Implied advective tendency of total energy path assuming closed budget",
    )
)
def implied_tendency_of_total_energy_ace2_path_due_to_advection(
    data: AtmosphereData,
    timestep: datetime.timedelta,
):
    """Implied tendency of total energy path due to advection.

    This is computed as a residual from the column total energy budget.
    """
    column_energy_tendency = total_energy_ace2_path_tendency(data, timestep)
    flux_through_vertical_boundaries = data.net_energy_flux_into_atmosphere
    implied_column_heating = column_energy_tendency - flux_through_vertical_boundaries
    return implied_column_heating


@register(VariableMetadata("m/s", "Windspeed at 10m above surface"))
def windspeed_at_10m(data: AtmosphereData, timestep: datetime.timedelta):
    return data.windspeed_at_10m


def _compute_derived_variable(
    data: TensorDict,
    vertical_coordinate: HasAtmosphereVerticalIntegral | None,
    timestep: datetime.timedelta,
    label: str,
    derived_variable_func: DerivedVariableFunc,
    forcing_data: TensorDict | None = None,
) -> TensorDict:
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

    atmosphere_data = AtmosphereData(data, vertical_coordinate)

    try:
        output = derived_variable_func(atmosphere_data, timestep)
    except KeyError as key_error:
        logging.debug(f"Could not compute {label} because {key_error} is missing")
    else:  # if no exception was raised
        new_data[label] = output
    return new_data


def compute_derived_quantities(
    data: TensorDict,
    vertical_coordinate: HasAtmosphereVerticalIntegral | None,
    timestep: datetime.timedelta,
    forcing_data: TensorDict | None = None,
) -> TensorDict:
    """Computes all derived quantities from the given data."""
    for label in _DERIVED_VARIABLE_REGISTRY:
        func = _DERIVED_VARIABLE_REGISTRY[label][0]
        data = _compute_derived_variable(
            data,
            vertical_coordinate,
            timestep,
            label,
            func,
            forcing_data=forcing_data,
        )
    return data
