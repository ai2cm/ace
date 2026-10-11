import datetime
import functools
import logging
from collections.abc import Callable, MutableMapping

import torch

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
    """Recover sea ice thickness (HI) in meters from sea ice volume in m^3.

    HI = sea_ice_volume / (area_weights_m2 * sea_ice_frac)
    """
    return data.sea_ice_thickness


@register(VariableMetadata("W/m**2", "Surface ocean heat flux"), exists_ok=True)
def hfds(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Compute the net downward surface heat flux."""
    return data.net_downward_surface_heat_flux


FROZEN_MASS_NAME = "frozen_mass_total_area"
ICE_LAYER_TEMPERATURE_NAMES = ["T1", "T2", "T3", "T4"]


class _FrozenEnthalpyTerms:
    """Frozen (ice + snow) enthalpy budget terms, per total cell area.

    With m frozen mass (kg/m**2), D its enthalpy deficit (J/kg) and k the
    time index (dim 1)::

        storage = (m(k) D(k) - m(k-1) D(k-1)) / dt,  storage[:, 0] = 0
        S       = hfds + lf_snowfl + minus_f_top + calving_residue
                  + minus_hfrunoffds
        r       = S - storage                          (unrestricted)
        E       = m(k-1) > 0 or m(k) > 0               (ice cells), E[:, 0] = False
        support = m(k-1) > 0 and m(k) > 0 and ssf > 0  (corrector S_h)

    S and D as in ``fme.core.corrector.ocean.SeaIceHfdsCorrection``. Terms are
    computed on access, so storage needs no flux inputs.
    """

    def __init__(self, data: OceanData, timestep: datetime.timedelta):
        self._data = data
        self._fields = data.data
        self._timestep_seconds = timestep.total_seconds()
        self._m = self._fields[FROZEN_MASS_NAME]

    @functools.cached_property
    def storage(self) -> torch.Tensor:
        # deferred: corrector.ocean -> dataset_info -> coordinates -> this module
        from fme.core.corrector.ocean import _frozen_deficit

        m = self._m
        mD = m * _frozen_deficit(self._fields, m, ICE_LAYER_TEMPERATURE_NAMES)
        storage = torch.zeros_like(mD)
        storage[:, 1:] = torch.diff(mD, n=1, dim=1) / self._timestep_seconds
        return storage

    @functools.cached_property
    def residual(self) -> torch.Tensor:
        from fme.core.corrector.ocean import SEA_ICE_FLUX_TERMS, surface_flux_term

        ssf = self._data.sea_surface_fraction
        source = self._data.net_downward_surface_heat_flux_total_area
        for term in SEA_ICE_FLUX_TERMS:
            # derive input merges gen and forcing, so both lookups read one dict
            source = source + surface_flux_term(term, self._fields, self._fields, ssf)
        return source - self.storage

    def _pair(self, combine) -> torch.Tensor:
        first = torch.zeros_like(self._m[:, :1], dtype=torch.bool)
        both = combine(self._m[:, :-1] > 0, self._m[:, 1:] > 0)
        return torch.cat([first, both], dim=1)

    @property
    def ice_cells(self) -> torch.Tensor:
        return self._pair(torch.logical_or)

    @property
    def support(self) -> torch.Tensor:
        return self._pair(torch.logical_and) & (self._data.sea_surface_fraction > 0)


def _frozen_enthalpy_terms(
    data: OceanData, timestep: datetime.timedelta
) -> _FrozenEnthalpyTerms:
    return _FrozenEnthalpyTerms(data, timestep)


@register(
    VariableMetadata(
        "W/m**2",
        "Frozen enthalpy budget residual on ice cells, per total cell area",
    )
)
def frozen_enthalpy_budget_residual(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """Residual r = S - d/dt(m D) where m(k-1) > 0 or m(k) > 0, else 0. Off those cells
    r is precipitation heat and snowfall mismatch, not a frozen budget term.
    """
    terms = _frozen_enthalpy_terms(data, timestep)
    return torch.where(terms.ice_cells, terms.residual, 0.0)


@register(
    VariableMetadata(
        "kg/m**2/s", "Tendency of frozen (ice + snow) mass per total cell area"
    )
)
def frozen_mass_tendency(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    m = data.data[FROZEN_MASS_NAME]
    tendency = torch.zeros_like(m)
    tendency[:, 1:] = torch.diff(m, n=1, dim=1) / timestep.total_seconds()
    return tendency


@register(
    VariableMetadata(
        "W/m**2",
        "Frozen enthalpy budget residual on freeze-up and melt-out cells, "
        "per total cell area",
    )
)
def frozen_enthalpy_budget_residual_off_support(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    """R on ice cells outside the sea-ice hfds corrector support, else 0: the
    part of r the corrector cannot close.
    """
    terms = _frozen_enthalpy_terms(data, timestep)
    return torch.where(terms.ice_cells & ~terms.support, terms.residual, 0.0)


@register(
    VariableMetadata(
        "W/m**2",
        "Tendency of frozen enthalpy deficit content per total cell area",
    )
)
def frozen_enthalpy_storage_tendency(
    data: OceanData,
    timestep: datetime.timedelta,
) -> torch.Tensor:
    return _frozen_enthalpy_terms(data, timestep).storage
