import datetime
import logging
from collections.abc import Callable, MutableMapping

import torch

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.typing_ import TensorDict

OceanDerivedVariableFunc = Callable[[OceanData, datetime.timedelta], torch.Tensor]

_OCEAN_DERIVED_VARIABLE_REGISTRY: MutableMapping[
    str, tuple[OceanDerivedVariableFunc, VariableMetadata, bool]
] = {}


def get_ocean_derived_variable_metadata() -> dict[str, VariableMetadata]:
    return {
        label: metadata
        for label, (_, metadata, _) in _OCEAN_DERIVED_VARIABLE_REGISTRY.items()
    }


def register(metadata: VariableMetadata, exists_ok: bool = False):
    def decorator(func: OceanDerivedVariableFunc):
        label = func.__name__
        if label in _OCEAN_DERIVED_VARIABLE_REGISTRY:
            raise ValueError(f"Function {label} has already been added to registry.")
        _OCEAN_DERIVED_VARIABLE_REGISTRY[label] = (func, metadata, exists_ok)
        return func

    return decorator


def _compute_ocean_derived_variable(
    data: TensorDict,
    depth_coordinate: HasOceanDepthIntegral | None,
    timestep: datetime.timedelta,
    label: str,
    derived_variable_func: OceanDerivedVariableFunc,
    forcing_data: TensorDict | None = None,
    exists_ok: bool = False,
) -> TensorDict:
    """Computes an ocean derived variable and adds it to the given data.

    By default the derived variable name must not already exist in the data,
    unless explicitly allowed with exists_ok=True.

    If any required input data are not available,
    the derived variable will not be computed.

    Args:
        data: dictionary of data to add the derived variable to.
        depth_coordinate: the depth coordinate.
        timestep: Timestep of the model.
        label: the name of the derived variable.
        derived_variable_func: derived variable function to compute.
        forcing_data: optional dictionary of forcing data needed for some derived
            variables. If necessary forcing inputs are missing, the derived
            variable will not be computed.
        exists_ok: Whether or not to allow the label to already exist in data,
            in which case a copy of the data TensorDict is returned with values
            unchanged.

    Returns:
        A new data dictionary with the derived variable added.
    """
    new_data = data.copy()
    if label in new_data:
        if exists_ok:
            return new_data
        raise ValueError(
            f"Variable {label} already exists. It is not permitted "
            "to have derived variables with same name as existing variables "
            "unless the derived variable is registered with exists_ok=True."
        )

    if forcing_data is not None:
        for key, value in forcing_data.items():
            if key not in data:
                data[key] = value

    ocean_data = OceanData(data, depth_coordinate)

    try:
        output = derived_variable_func(ocean_data, timestep)
    except KeyError as key_error:
        logging.debug(f"Could not compute {label} because {key_error} is missing")
    else:  # if no exception was raised
        new_data[label] = output
    return new_data


def compute_ocean_derived_quantities(
    data: TensorDict,
    depth_coordinate: HasOceanDepthIntegral | None,
    timestep: datetime.timedelta,
    forcing_data: TensorDict | None = None,
) -> TensorDict:
    """Computes all derived quantities from the given data."""
    for label in _OCEAN_DERIVED_VARIABLE_REGISTRY:
        func = _OCEAN_DERIVED_VARIABLE_REGISTRY[label][0]
        exists_ok = _OCEAN_DERIVED_VARIABLE_REGISTRY[label][2]
        data = _compute_ocean_derived_variable(
            data,
            depth_coordinate,
            timestep,
            label,
            func,
            forcing_data=forcing_data,
            exists_ok=exists_ok,
        )
    return data


@register(VariableMetadata("J/m**2", "Column-integrated ocean heat content"))
def ocean_heat_content(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean heat content."""
    return data.ocean_heat_content


@register(VariableMetadata("[0-1]", "sea ice concentration"), exists_ok=True)
def sea_ice_fraction(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the sea ice fraction."""
    return data.sea_ice_fraction
