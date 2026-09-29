"""Frozen-mass (sea ice + snow) energy budget terms.

Shared by the data loader (``fme.core.dataset.derived``), which derives
``frozen_mass``, ``calving_residue`` and ``frozen_mass_energy_budget_residual``
from stored fields, and by the ocean corrector (``fme.core.corrector.ocean``),
which applies the surface energy flux correction and the frozen-mass budget at
step time. Both call the same functions, so the loader's residual target and
the corrector's step-time flux sum are built identically.

Area bases:
    - ``frozen_mass`` and the flux sum are per total cell area:
      ``fc(x) = sea_surface_fraction * nan_to_num(x)``.
    - ``calving_residue`` and ``hfrunoffds`` are per ocean area.

This module must not import from ``fme.core.corrector`` or ``fme.core.dataset``.
"""

from collections.abc import Mapping
from typing import Literal

import torch

from fme.core.atmosphere_data import AtmosphereData
from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_FREEZING,
    LATENT_HEAT_OF_VAPORIZATION,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.ocean_data import OceanData
from fme.core.typing_ import TensorMapping

HfdsCorrectionMethod = Literal[
    "residual_prediction", "prescribed", "prescribed_open_ocean"
]


def _fc(x: torch.Tensor, sea_surface_fraction: torch.Tensor) -> torch.Tensor:
    """Per ocean area -> per total cell area, with NaN (land) set to 0."""
    return torch.nan_to_num(sea_surface_fraction) * torch.nan_to_num(x)


def ocean_net_surface_energy_flux(
    forcing_data: TensorMapping,
    sst: torch.Tensor,
) -> torch.Tensor:
    """Compute the net surface energy flux into the ocean from atmospheric
    forcing variables and the sea surface temperature.

    This extends the atmosphere net surface energy flux with SST-dependent
    heat transport by precipitation and evaporation.
    """
    atmos = AtmosphereData(forcing_data)
    base_flux = (
        atmos.net_surface_energy_flux
    )  # missing: - calving * LATENT_HEAT_OF_FREEZING
    mass_heat_flux = (
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * (
            atmos.precipitation_rate
            + atmos.frozen_precipitation_rate
            - (atmos.latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION)
        )  # missing: + river runoff + calving
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    )
    return base_flux + mass_heat_flux


def correct_hfds(
    net_flux: torch.Tensor,
    gen_hfds: torch.Tensor,
    ocean_fraction: torch.Tensor,
    method: HfdsCorrectionMethod,
    sea_surface_fraction: torch.Tensor | None = None,
    hfrunoffds: torch.Tensor | None = None,
    calving_residue: torch.Tensor | None = None,
) -> torch.Tensor:
    """Corrected hfds from the per-ocean-area ``net_flux``.

    If ``hfrunoffds`` and ``calving_residue`` are given, ``net_flux`` gains
    ``+ hfrunoffds - calving_residue`` (per ocean area). If
    ``sea_surface_fraction`` is given, ``gen_hfds`` is ``hfds_total_area`` and
    ``net_flux`` is multiplied by it after the runoff and calving terms.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
        prescribed_open_ocean: net_flux where ocean_fraction == 1, else gen_hfds
    """
    if (hfrunoffds is None) != (calving_residue is None):
        raise ValueError(
            "hfrunoffds and calving_residue must be given together, got "
            f"hfrunoffds={hfrunoffds is not None}, "
            f"calving_residue={calving_residue is not None}"
        )
    if hfrunoffds is not None and calving_residue is not None:
        net_flux = net_flux + hfrunoffds - calving_residue
    if sea_surface_fraction is not None:
        net_flux = net_flux * sea_surface_fraction
    if method == "residual_prediction":
        return net_flux * ocean_fraction + gen_hfds
    elif method == "prescribed":
        return net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    elif method == "prescribed_open_ocean":
        return torch.where(ocean_fraction == 1, net_flux, gen_hfds)
    raise NotImplementedError(
        f"Method {method!r} not implemented for surface energy flux correction"
    )


def frozen_mass(
    simass: torch.Tensor,
    sisnmass: torch.Tensor,
    sea_surface_fraction: torch.Tensor,
) -> torch.Tensor:
    """Sea ice plus snow mass per total cell area [kg m-2].

    ``frozen_mass = fc(simass + sisnmass)``.
    """
    return _fc(
        torch.nan_to_num(simass) + torch.nan_to_num(sisnmass), sea_surface_fraction
    )


def calving_residue(
    hflso: torch.Tensor,
    evs: torch.Tensor,
    prsn: torch.Tensor,
) -> torch.Tensor:
    """Per-ocean-area residue [W m-2]:
    ``nan_to_num(-hflso + L_v evs - L_f prsn)``.
    """
    return torch.nan_to_num(
        -hflso + LATENT_HEAT_OF_VAPORIZATION * evs - LATENT_HEAT_OF_FREEZING * prsn
    )


def frozen_mass_flux_sum(
    forcing_data: TensorMapping,
    hfds_total_area: torch.Tensor,
    hfrunoffds: torch.Tensor,
    calving_residue: torch.Tensor,
) -> torch.Tensor:
    """Step-time energy flux sum ``S`` into the frozen mass [W m-2].

    Per total cell area:

        S = fc(L_f P_snow - F_top) + hfds_total_area - fc(hfrunoffds)
            + fc(calving_residue)

    where ``L_f P_snow - F_top = -AtmosphereData(forcing_data).net_surface_energy_flux``
    and ``fc`` multiplies by ``OceanData(forcing_data).sea_surface_fraction``.
    ``hfds_total_area`` is the corrected one (``correct_hfds`` with
    ``sea_surface_fraction``).
    """
    ssf = OceanData(forcing_data).sea_surface_fraction
    atmos_flux = AtmosphereData(forcing_data).net_surface_energy_flux
    return (
        _fc(-atmos_flux, ssf)
        + torch.nan_to_num(hfds_total_area)
        - _fc(hfrunoffds, ssf)
        + _fc(calving_residue, ssf)
    )


def frozen_mass_energy_budget_residual(
    flux_sum: torch.Tensor,
    frozen_mass: torch.Tensor,
    frozen_mass_previous: torch.Tensor,
    timestep_seconds: float,
) -> torch.Tensor:
    """Budget residual [W m-2]:
    ``r = S - L_f (frozen_mass - frozen_mass_previous) / dt``.
    """
    return (
        flux_sum
        - LATENT_HEAT_OF_FREEZING
        * (frozen_mass - frozen_mass_previous)
        / timestep_seconds
    )


def target_frozen_mass_energy_budget_residual(
    data: Mapping[str, torch.Tensor],
    timestep_seconds: float,
    hfds_method: HfdsCorrectionMethod = "prescribed",
) -> torch.Tensor:
    """The residual on stored target data of one window, time on dim 0.

    Index ``k >= 1`` is built as the corrector builds a step from ``k-1`` to
    ``k``: ``sst`` and ``ocean_fraction`` from index ``k-1`` (the step input),
    the atmosphere fluxes, ``hfds_total_area``, ``hfrunoffds`` and the fluxes
    of ``calving_residue`` from index ``k``:

        hfds_c(k) = correct_hfds(net_flux(k; sst(k-1)), hfds_total_area(k),
                                 ocean_fraction(k-1), hfds_method, ssf,
                                 hfrunoffds(k), calving_residue(k))
        r(k)      = frozen_mass_flux_sum(k; hfds_c(k))
                    - L_f (frozen_mass(k) - frozen_mass(k-1)) / dt

    Index 0 needs ``k = -1``, outside the window, and is set to 0. It is never
    scored: out-only names are not in the initial condition.

    Computed in float64 and returned in the dtype of ``data["simass"]``.
    """
    dtype = data["simass"].dtype
    d = {k: torch.nan_to_num(v.to(torch.float64)) for k, v in data.items()}
    m = frozen_mass(d["simass"], d["sisnmass"], d["sea_surface_fraction"])
    cr = calving_residue(d["hflso"], d["evs"], d["prsn"])
    previous = {k: v[:-1] for k, v in d.items()}
    current = {k: v[1:] for k, v in d.items()}
    previous_ocean = OceanData(previous)
    net_flux = ocean_net_surface_energy_flux(
        current, previous_ocean.sea_surface_temperature
    )
    hfds_c = correct_hfds(
        net_flux,
        current["hfds_total_area"],
        previous_ocean.ocean_fraction,
        hfds_method,
        sea_surface_fraction=current["sea_surface_fraction"],
        hfrunoffds=current["hfrunoffds"],
        calving_residue=cr[1:],
    )
    s = frozen_mass_flux_sum(current, hfds_c, current["hfrunoffds"], cr[1:])
    r = torch.zeros_like(m)
    r[1:] = frozen_mass_energy_budget_residual(s, m[1:], m[:-1], timestep_seconds)
    return r.to(dtype)
