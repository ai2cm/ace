import datetime
import logging
from typing import Callable, Dict, MutableMapping, Optional, Tuple

import torch

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.typing_ import TensorDict

OceanDerivedVariableFunc = Callable[[OceanData, datetime.timedelta], torch.Tensor]

UNMASKED_OCEAN_DERIVED_NAMES = ["ocean_heat_content"]

_OCEAN_DERIVED_VARIABLE_REGISTRY: MutableMapping[
    str, Tuple[OceanDerivedVariableFunc, VariableMetadata]
] = {}


def get_ocean_derived_variable_metadata() -> Dict[str, VariableMetadata]:
    return {
        label: metadata
        for label, (_, metadata) in _OCEAN_DERIVED_VARIABLE_REGISTRY.items()
    }


def register(metadata: VariableMetadata):
    def decorator(func: OceanDerivedVariableFunc):
        label = func.__name__
        if label in _OCEAN_DERIVED_VARIABLE_REGISTRY:
            raise ValueError(f"Function {label} has already been added to registry.")
        _OCEAN_DERIVED_VARIABLE_REGISTRY[label] = (func, metadata)
        return func

    return decorator


def _compute_ocean_derived_variable(
    data: TensorDict,
    depth_coordinate: Optional[HasOceanDepthIntegral],
    timestep: datetime.timedelta,
    label: str,
    derived_variable_func: OceanDerivedVariableFunc,
    forcing_data: Optional[TensorDict] = None,
) -> TensorDict:
    """Computes an ocean derived variable and adds it to the given data.

    The derived variable name must not already exist in the data.

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
    depth_coordinate: Optional[HasOceanDepthIntegral],
    timestep: datetime.timedelta,
    forcing_data: Optional[TensorDict] = None,
) -> TensorDict:
    """Computes all derived quantities from the given data."""
    for label in _OCEAN_DERIVED_VARIABLE_REGISTRY:
        func = _OCEAN_DERIVED_VARIABLE_REGISTRY[label][0]
        data = _compute_ocean_derived_variable(
            data,
            depth_coordinate,
            timestep,
            label,
            func,
            forcing_data=forcing_data,
        )
    return data


@register(VariableMetadata("J/m**2", "Column-integrated ocean heat content"))
def ocean_heat_content(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean heat content."""
    return data.ocean_heat_content
