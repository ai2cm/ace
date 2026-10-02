import datetime
import logging
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING

import torch

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.ocean_data import (
    LAYER_OHC_DEFAULT_BANDS,
    HasCellAreaInMetersSquared,
    HasOceanDepthIntegral,
    OceanData,
    layer_ohc_name,
)
from fme.core.typing_ import TensorDict

if TYPE_CHECKING:
    from fme.core.coordinates import VerticalCoordinate

OceanDerivedVariableFunc = Callable[[OceanData, datetime.timedelta], torch.Tensor]
OceanMultiDerivedVariableFunc = Callable[[OceanData, datetime.timedelta], TensorDict]

_OCEAN_DERIVED_VARIABLE_REGISTRY: MutableMapping[
    str, tuple[OceanDerivedVariableFunc, VariableMetadata, bool]
] = {}

# label -> (func, metadata of the output names known before computing)
_OCEAN_MULTI_DERIVED_VARIABLE_REGISTRY: MutableMapping[
    str, tuple[OceanMultiDerivedVariableFunc, dict[str, VariableMetadata]]
] = {}


def get_ocean_derived_variable_metadata() -> dict[str, VariableMetadata]:
    metadata = {
        label: metadata
        for label, (_, metadata, _) in _OCEAN_DERIVED_VARIABLE_REGISTRY.items()
    }
    for _, names_metadata in _OCEAN_MULTI_DERIVED_VARIABLE_REGISTRY.values():
        metadata.update(names_metadata)
    return metadata


def register(metadata: VariableMetadata, exists_ok: bool = False):
    def decorator(func: OceanDerivedVariableFunc):
        label = func.__name__
        if label in _OCEAN_DERIVED_VARIABLE_REGISTRY:
            raise ValueError(f"Function {label} has already been added to registry.")
        _OCEAN_DERIVED_VARIABLE_REGISTRY[label] = (func, metadata, exists_ok)
        return func

    return decorator


def register_multi(metadata: dict[str, VariableMetadata]):
    """Register a function returning several derived variables, keyed by name.

    Args:
        metadata: Metadata of those output names known at registration; names
            that depend on the data (e.g. one per depth level) may be absent.
    """

    def decorator(func: OceanMultiDerivedVariableFunc):
        label = func.__name__
        if (
            label in _OCEAN_DERIVED_VARIABLE_REGISTRY
            or label in _OCEAN_MULTI_DERIVED_VARIABLE_REGISTRY
        ):
            raise ValueError(f"Function {label} has already been added to registry.")
        _OCEAN_MULTI_DERIVED_VARIABLE_REGISTRY[label] = (func, metadata)
        return func

    return decorator


def _compute_ocean_multi_derived_variable(
    data: TensorDict,
    depth_coordinate: HasOceanDepthIntegral | None,
    timestep: datetime.timedelta,
    label: str,
    func: OceanMultiDerivedVariableFunc,
    cell_area_provider: HasCellAreaInMetersSquared | None = None,
) -> TensorDict:
    """``data`` with the outputs of ``func`` added; unchanged if an input is
    missing. No output name may already exist in ``data``.
    """
    ocean_data = OceanData(
        data, depth_coordinate, cell_area_provider=cell_area_provider
    )
    try:
        output = func(ocean_data, timestep)
    except KeyError as key_error:
        logging.debug(f"Could not compute {label} because {key_error} is missing")
        return data
    existing = sorted(set(output).intersection(data))
    if existing:
        raise ValueError(
            f"Variables {existing} of {label} already exist. It is not permitted "
            "to have derived variables with same name as existing variables."
        )
    return {**data, **output}


def _compute_ocean_derived_variable(
    data: TensorDict,
    depth_coordinate: HasOceanDepthIntegral | None,
    timestep: datetime.timedelta,
    label: str,
    derived_variable_func: OceanDerivedVariableFunc,
    forcing_data: TensorDict | None = None,
    exists_ok: bool = False,
    cell_area_provider: HasCellAreaInMetersSquared | None = None,
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
        cell_area_provider: optional provider of cell areas in meters squared,
            needed by some derived variables.

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

    ocean_data = OceanData(
        data, depth_coordinate, cell_area_provider=cell_area_provider
    )

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
    cell_area_provider: HasCellAreaInMetersSquared | None = None,
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
            cell_area_provider=cell_area_provider,
        )
    for label, (multi_func, _) in _OCEAN_MULTI_DERIVED_VARIABLE_REGISTRY.items():
        data = _compute_ocean_multi_derived_variable(
            data,
            depth_coordinate,
            timestep,
            label,
            multi_func,
            cell_area_provider=cell_area_provider,
        )
    return data


@register(VariableMetadata("J/m**2", "Column-integrated ocean heat content"))
def ocean_heat_content(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean heat content."""
    return data.ocean_heat_content


@register(
    VariableMetadata("W/m**2", "Tendency of column-integrated ocean heat content")
)
def ocean_heat_content_tendency(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean heat content tendency."""
    ohc = data.ocean_heat_content
    ohc_tendency = torch.zeros_like(ohc)
    ohc_tendency[:, 1:] = torch.diff(ohc, n=1, dim=1) / timestep.total_seconds()
    return ohc_tendency


@register(
    VariableMetadata(
        "W/m**2",
        "Implied advective tendency of ocean heat content assuming closed budget",
    )
)
def implied_tendency_of_ocean_heat_content_due_to_advection(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Implied tendency of ocean heat content due to advection.
    This is computed as a residual from the column total energy budget.
    """
    column_energy_tendency = ocean_heat_content_tendency(data, timestep)
    flux_through_vertical_boundaries = data.net_energy_flux_into_ocean
    implied_column_heating = column_energy_tendency - flux_through_vertical_boundaries
    return implied_column_heating


@register(
    VariableMetadata(
        "W/m**2",
        "Net energy flux through surface and sea floor into ocean",
    )
)
def net_energy_flux_into_ocean_column(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.net_energy_flux_into_ocean


@register(VariableMetadata("m", "Mixed layer depth, Wright (1997) density threshold"))
def mld_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Density-threshold mixed layer depth, positive down."""
    return data.mld_wright97


@register_multi({})
def rho_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> TensorDict:
    """``rho_wright97_{k}``, Wright (1997) in-situ density anomaly [kg/m**3]."""
    return data.rho_wright97


@register(
    VariableMetadata("Pa", "Globally demeaned bottom pressure anomaly, Wright (1997)")
)
def pbo_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.pbo_wright97


@register(VariableMetadata("m", "Globally demeaned steric height, Wright (1997)"))
def steric_height_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.steric_height_wright97


@register_multi(
    {
        layer_ohc_name(band): VariableMetadata(
            "J/m**2",
            f"Ocean heat content from {band[0]:g} m to "
            + ("the sea floor" if band[1] is None else f"{band[1]:g} m"),
        )
        for band in LAYER_OHC_DEFAULT_BANDS
    }
)
def layer_ohc(
    data: OceanData,
    timestep: datetime.timedelta,
) -> TensorDict:
    """``layer_ohc_{a}_{b}`` on ``LAYER_OHC_DEFAULT_BANDS``."""
    return data.layer_ohc


def ocean_derived_spatial_masks(
    depth_coordinate: "VerticalCoordinate",
) -> dict[str, torch.Tensor]:
    """``mask_<name>`` for each registered derived variable whose NaN pattern
    no data mask matches (``layer_ohc_*``: the cells where the band exists),
    for the aggregators' spatial mask provider; empty without a
    ``DepthCoordinate``.
    """
    from fme.core.coordinates import DepthCoordinate
    from fme.core.optimized_derived import layer_ohc_derivation

    # layer_ohc needs idepth, mask and dz, which only DepthCoordinate has
    if not isinstance(depth_coordinate, DepthCoordinate):
        return {}
    derivation = layer_ohc_derivation(depth_coordinate)
    return {} if derivation is None else derivation.spatial_masks


@register(VariableMetadata("[0-1]", "sea ice concentration"), exists_ok=True)
def sea_ice_fraction(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the sea ice fraction."""
    return data.sea_ice_fraction


@register(VariableMetadata("m", "Sea ice thickness"), exists_ok=True)
def HI(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Recover sea ice thickness (HI) in meters from sea ice volume in 1000 * km^3.

    HI = sea_ice_volume * 1e9 / (area_weights_m2 * sea_ice_frac)
    """
    return data.sea_ice_thickness


@register(VariableMetadata("W/m**2", "Surface ocean heat flux"), exists_ok=True)
def hfds(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the net downward surface heat flux."""
    return data.net_downward_surface_heat_flux
