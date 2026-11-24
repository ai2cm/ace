import datetime
import logging
from collections.abc import Callable, MutableMapping

import torch

from fme.core.area_weights_provider import (
    AreaWeightsProviderABC,
    NullAreaWeightsProvider,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.ocean_data import (
    HasCellAreaInMetersSquared,
    HasOceanDepthIntegral,
    OceanData,
)
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
    cell_area_provider: HasCellAreaInMetersSquared | None = None,
    area_weights_provider: AreaWeightsProviderABC = NullAreaWeightsProvider,
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
        area_weights_provider: the dataset's ocean-area weights, used by
            derived variables that take ocean-area means.

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
        data,
        depth_coordinate,
        cell_area_provider=cell_area_provider,
        area_weights_provider=area_weights_provider,
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
    area_weights_provider: AreaWeightsProviderABC = NullAreaWeightsProvider,
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
            area_weights_provider=area_weights_provider,
        )
    return data


@register(
    VariableMetadata(
        "J/m**2", "Column-integrated ocean heat content per unit ocean area"
    )
)
def ocean_heat_content(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean heat content."""
    return data.ocean_heat_content


@register(
    VariableMetadata(
        "W/m**2",
        "Tendency of column-integrated ocean heat content per unit ocean area",
    )
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
        "Implied advective tendency of ocean heat content assuming closed budget, "
        "per unit total cell area",
    )
)
def implied_tendency_of_ocean_heat_content_due_to_advection(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Implied tendency of ocean heat content due to advection.
    This is computed as a residual from the column total energy budget, per
    unit total cell area: the ocean heat content tendency, which is per unit
    ocean area, is weighted by the sea surface fraction to match the net
    energy flux into the ocean.
    """
    column_energy_tendency = (
        ocean_heat_content_tendency(data, timestep) * data.sea_surface_fraction
    )
    flux_through_vertical_boundaries = data.net_energy_flux_into_ocean
    implied_column_heating = column_energy_tendency - flux_through_vertical_boundaries
    return implied_column_heating


@register(VariableMetadata("g/m**2", "Column-integrated ocean salt content"))
def ocean_salt_content(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean salt content."""
    return data.ocean_salt_content


@register(
    VariableMetadata("g/m**2/s", "Tendency of column-integrated ocean salt content")
)
def ocean_salt_content_tendency(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the column-integrated ocean salt content tendency."""
    osc = data.ocean_salt_content
    osc_tendency = torch.zeros_like(osc)
    osc_tendency[:, 1:] = torch.diff(osc, n=1, dim=1) / timestep.total_seconds()
    return osc_tendency


@register(
    VariableMetadata(
        "g/m**2/s",
        "Implied advective tendency of ocean salt content assuming closed budget",
    )
)
def implied_tendency_of_ocean_salt_content_due_to_advection(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Implied tendency of ocean salt content due to advection.
    This is computed as a residual from the column salt budget. A missing sea
    ice salt flux is taken as zero, in which case net_salt_flux_into_ocean_column
    is not computed. The missing brine regection term is globally small, and this
    implied tendancy is locally noisier than the net salt flux term.
    """
    column_salt_tendency = ocean_salt_content_tendency(data, timestep)
    try:
        flux_through_surface = data.net_salt_flux_into_ocean
    except KeyError:
        flux_through_surface = data.net_virtual_salt_flux_into_ocean
    return column_salt_tendency - flux_through_surface


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


@register(VariableMetadata("g/m**2/s", "Virtual salt flux through surface into ocean"))
def net_virtual_salt_flux_into_ocean_column(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.net_virtual_salt_flux_into_ocean


@register(VariableMetadata("g/m**2/s", "Net salt flux through surface into ocean"))
def net_salt_flux_into_ocean_column(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return data.net_salt_flux_into_ocean


@register(
    VariableMetadata("m", "Sea surface height minus its ocean-area global mean"),
    exists_ok=True,
)
def zos(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Sea surface height above geoid (zos) from sea surface height (SSH).

    MOM6 in the CM4 configuration (no surface pressure in SSH) writes

        zos_i = SSH_i - <SSH>_native

    with <x>_native the ocean-area-weighted global mean on the native grid
    (i: native cell). Here the mean is taken on the data grid (J: grid cell):

        zos_J = SSH_J - <SSH>_W,  <x>_W = sum_J(W_J x_J) / sum_J(W_J)
        W_J = ocean_area_weights_J * isfinite(SSH_J)
            if the dataset has an area weights provider (areacello * ssf), else
        W_J = area_weights_m2_J * sea_surface_fraction_J * isfinite(SSH_J)

    <SSH>_W equals <SSH>_native only when W preserves the native ocean area
    of each cell. So zos is exact only with an area weights provider (the
    store holds areacello and sea_surface_fraction). Without one, W uses
    cos-lat cell area (area_weights_m2), and zos differs from the native zos
    by a spatially uniform offset per sample. <zos>_W is zero by construction.

    Weights are on the global (lat, lon) grid. Spatial parallelism is not
    supported: the weights' shape does not match a local shard (same as HI).

    Registered before derived variables that read zos, and derived only when
    zos is absent from the data.
    """
    ssh = data.sea_surface_height
    ocean_area = data.ocean_area_weights
    if ocean_area is None:
        sea_surface_fraction = data.sea_surface_fraction
        try:
            cell_area = data.area_weights_m2
        except ValueError as err:
            raise KeyError("area_weights_m2") from err
        if cell_area is None:
            raise KeyError("area_weights_m2")
        ocean_area = cell_area * sea_surface_fraction
    finite = torch.isfinite(ssh)
    weights = torch.where(finite, ocean_area, 0.0)
    spatial_dims = (-2, -1)
    mean = (weights * torch.where(finite, ssh, 0.0)).sum(
        dim=spatial_dims, keepdim=True
    ) / weights.sum(dim=spatial_dims, keepdim=True)
    return ssh - mean


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
    """Sea ice thickness (HI) in meters, per unit sea ice area.

    Stored HI if present, else, per cell:

    HI = ocean_sea_ice_thickness_mean / ocean_sea_ice_fraction  (0 where the
    fraction is 0), else
    HI = sea_ice_volume / (area_weights_m2 * sea_ice_frac)

    where sea_ice_frac = ocean_sea_ice_fraction * sea_surface_fraction (per
    cell area).
    """
    return data.sea_ice_thickness


@register(VariableMetadata("m^3", "Sea ice volume"), exists_ok=True)
def sea_ice_volume(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Sea ice volume in the cell, in m^3.

    Stored sea_ice_volume if present, else, per cell:

    sea_ice_volume = ocean_sea_ice_thickness_mean * area_weights_m2
                     * sea_surface_fraction
                   = HI * ocean_sea_ice_fraction * ocean area

    0 where ocean_sea_ice_fraction is 0. Inverse of the volume path of HI.
    """
    return data.sea_ice_volume


@register(VariableMetadata("W/m**2", "Surface ocean heat flux"), exists_ok=True)
def hfds(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the net downward surface heat flux."""
    return data.net_downward_surface_heat_flux


@register(VariableMetadata("m", "Mixed layer depth, Wright (1997) density threshold"))
def mld_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Density-threshold mixed layer depth, positive down."""
    return data.mld_wright97


@register(VariableMetadata("m", "Steric sea level of the static column, Wright (1997)"))
def steric_sea_level_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Steric sea level of the static column, -C / RHO_0."""
    return data.steric_sea_level_wright97


@register(
    VariableMetadata("m", "Globally demeaned manometric sea level, Wright (1997)")
)
def manometric_sea_level_wright97(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Globally demeaned manometric sea level, zos minus steric sea level."""
    return data.manometric_sea_level_wright97
