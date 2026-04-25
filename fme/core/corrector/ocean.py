import dataclasses
import datetime
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import torch

from fme.core.atmosphere_data import ATMOSPHERE_FIELD_NAME_PREFIXES, AtmosphereData
from fme.core.constants import (
    FREEZING_TEMPERATURE_KELVIN,
    LATENT_HEAT_OF_FREEZING,
    LATENT_HEAT_OF_VAPORIZATION,
    SPECIFIC_HEAT_OF_SEA_WATER_CM4,
)
from fme.core.corrector.registry import CorrectorABC, CorrectorConfigABC
from fme.core.corrector.utils import force_positive
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import GriddedOperations
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class SeaIceFractionConfig:
    """Correct predicted sea_ice_fraction to ensure it is always in 0-1, and
    land_fraction + sea_ice_fraction + ocean_fraction = 1. After
    sea_ice_fraction is corrected, all variables listed in
    zero_where_ice_free_names will be set to 0 everywhere
    sea_ice_fraction is 0.

    Parameters:
        sea_ice_fraction_name: Name of the sea ice fraction variable.
        land_fraction_name: Name of the land fraction variable.
        zero_where_ice_free_names: List of variable names to set to 0
            wherever sea_ice_fraction is 0.
        remove_negative_ocean_fraction: If True, reduce sea_ice_fraction
            to prevent ocean_fraction (1 - sea_ice_fraction - land_fraction)
            from being negative.
    """

    sea_ice_fraction_name: str
    land_fraction_name: str
    zero_where_ice_free_names: list[str] = dataclasses.field(default_factory=list)
    remove_negative_ocean_fraction: bool = True

    def __call__(
        self, gen_data: TensorMapping, input_data: TensorMapping
    ) -> TensorDict:
        out = {**gen_data}
        out[self.sea_ice_fraction_name] = torch.clamp(
            out[self.sea_ice_fraction_name], min=0.0, max=1.0
        )
        if self.remove_negative_ocean_fraction:
            negative_ocean_fraction = (
                1
                - out[self.sea_ice_fraction_name]
                - input_data[self.land_fraction_name]
            )
            negative_ocean_fraction = negative_ocean_fraction.clip(max=0)
            out[self.sea_ice_fraction_name] += negative_ocean_fraction
        for name in self.zero_where_ice_free_names:
            out[name] = gen_data[name] * (out[self.sea_ice_fraction_name] > 0.0)
        return out


@dataclasses.dataclass
class OceanHeatContentBudgetConfig:
    """Configuration for ocean heat content budget correction.

    Parameters:
        method: Method to use for OHC budget correction. The available option is
            "scaled_temperature", which enforces conservation of heat content
            by scaling the predicted potential temperature by a vertically and
            horizontally uniform correction factor.
        constant_unaccounted_heating: Area-weighted global mean
            column-integrated heating in W/m**2 to be added to the energy flux
            into the ocean when conserving the heat content. This can be useful
            for correcting errors in heat budget in target data. The same
            additional heating is imposed at all time steps and grid cells.

    """

    method: Literal["scaled_temperature"]
    constant_unaccounted_heating: float = 0.0


# Atomic contributions to the net surface energy flux into the ocean. Each
# entry is a callable taking (AtmosphereData, sst) and returning a tensor with
# the signed contribution of that atomic term in W/m^2.
_ATOMIC_CONTRIBUTIONS: dict[
    str, Callable[[AtmosphereData, torch.Tensor], torch.Tensor]
] = {
    "sw_down": lambda atmos, sst: atmos.sfc_down_sw_radiative_flux,
    "sw_up": lambda atmos, sst: -atmos.sfc_up_sw_radiative_flux,
    "lw_down": lambda atmos, sst: atmos.sfc_down_lw_radiative_flux,
    "lw_up": lambda atmos, sst: -atmos.sfc_up_lw_radiative_flux,
    "lhf_turbulent": lambda atmos, sst: -atmos.latent_heat_flux,
    "shf_turbulent": lambda atmos, sst: -atmos.sensible_heat_flux,
    "frozen_precip_latent": (
        lambda atmos, sst: -LATENT_HEAT_OF_FREEZING * atmos.frozen_precipitation_rate
    ),
    "lhf_mass": lambda atmos, sst: -(
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * (atmos.latent_heat_flux / LATENT_HEAT_OF_VAPORIZATION)
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    ),
    "precip_mass": lambda atmos, sst: (
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * atmos.precipitation_rate
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    ),
    "frozen_precip_mass": lambda atmos, sst: (
        SPECIFIC_HEAT_OF_SEA_WATER_CM4
        * atmos.frozen_precipitation_rate
        * (sst - FREEZING_TEMPERATURE_KELVIN)
    ),
}

# Mapping from a user-facing flux variable / bucket name to the set of atomic
# contributions it implies. The atomic decomposition lets the user combine
# granular variable names (e.g. "latent_heat_flux") with bucket names (e.g.
# "net_surface_energy_flux") without double-counting overlapping terms.
_NAME_TO_ATOMIC_CONTRIBUTIONS: dict[str, frozenset[str]] = {
    "sfc_down_sw_radiative_flux": frozenset({"sw_down"}),
    "sfc_up_sw_radiative_flux": frozenset({"sw_up"}),
    "sfc_down_lw_radiative_flux": frozenset({"lw_down"}),
    "sfc_up_lw_radiative_flux": frozenset({"lw_up"}),
    "latent_heat_flux": frozenset({"lhf_turbulent", "lhf_mass"}),
    "sensible_heat_flux": frozenset({"shf_turbulent"}),
    "precipitation_rate": frozenset({"precip_mass"}),
    "frozen_precipitation_rate": frozenset(
        {"frozen_precip_latent", "frozen_precip_mass"}
    ),
    "net_surface_energy_flux": frozenset(
        {
            "sw_down",
            "sw_up",
            "lw_down",
            "lw_up",
            "lhf_turbulent",
            "shf_turbulent",
            "frozen_precip_latent",
        }
    ),
    "net_surface_energy_flux_without_frozen_precip": frozenset(
        {"sw_down", "sw_up", "lw_down", "lw_up", "lhf_turbulent", "shf_turbulent"}
    ),
}

# Atmosphere flux variables that are valid `names` entries but whose key in
# ATMOSPHERE_FIELD_NAME_PREFIXES does not contain "surface" or "sfc".
_VALID_FLUX_NAMES_NOT_MATCHING_PATTERN = [
    "latent_heat_flux",
    "sensible_heat_flux",
    "precipitation_rate",
    "frozen_precipitation_rate",
]

# AtmosphereData @property names that contain "surface" or "sfc" — discovered at
# import so that conceptual buckets like net_surface_energy_flux are accepted in
# `names` automatically when added to AtmosphereData.
_ATMOSPHERE_DATA_SURFACE_PROPERTIES = frozenset(
    name
    for name, attr in vars(AtmosphereData).items()
    if isinstance(attr, property) and ("surface" in name or "sfc" in name)
)


@dataclasses.dataclass
class SurfaceEnergyFluxCorrectionConfig:
    """Configuration for correcting the generated hfds using
    atmosphere-derived surface energy fluxes and ocean_fraction.

    The net_flux is the net surface energy flux computed from atmospheric
    forcing variables and generated SST. The ocean_fraction naturally zeroes
    out the correction on land and reduces it under sea ice.

    Available options are:
      - "residual_prediction": corrected_hfds = gen_hfds + ocean_fraction * net_flux.
        The network predicts a residual that is added to the forcing-derived flux.
      - "prescribed": corrected_hfds = net_flux * ocean_fraction + gen_hfds *
        (1 - ocean_fraction). Open-ocean hfds is prescribed from forcings; the
        network prediction is retained under sea ice and on land.

    Parameters:
        method: Method to use for the correction.
        names: Atmosphere flux terms to include in the net_flux computation.
            "default" (the default) uses the legacy hard-coded combination of
            sw/lw radiative fluxes, latent and sensible heat fluxes, frozen
            precipitation latent heat, and the sst-dependent mass-heat fluxes
            from precipitation, frozen precipitation, and evaporation. A list
            of strings selects a specific subset; each name is decomposed into
            atomic contributions (e.g. "latent_heat_flux" implies both the
            turbulent term and the implicit evaporation mass-heat term) and
            overlapping atomic contributions across listed names are summed
            once. Valid names are keys of ATMOSPHERE_FIELD_NAME_PREFIXES that
            contain "surface" or "sfc", a small private list of additional
            atmosphere flux variables, and AtmosphereData @property names that
            contain "surface" or "sfc" (e.g. "net_surface_energy_flux").

    """

    method: Literal["residual_prediction", "prescribed"]
    names: Literal["default"] | list[str] = "default"

    def __post_init__(self):
        if self.names == "default":
            return
        allow_set = (
            {k for k in ATMOSPHERE_FIELD_NAME_PREFIXES if "surface" in k or "sfc" in k}
            | set(_VALID_FLUX_NAMES_NOT_MATCHING_PATTERN)
            | _ATMOSPHERE_DATA_SURFACE_PROPERTIES
        )
        unknown = [n for n in self.names if n not in allow_set]
        if unknown:
            raise ValueError(
                f"Surface flux name(s) {unknown} are not recognized atmospheric "
                f"surface variables. Allowed: {sorted(allow_set)}"
            )
        unsupported = [n for n in self.names if n not in _NAME_TO_ATOMIC_CONTRIBUTIONS]
        if unsupported:
            raise ValueError(
                f"Surface flux name(s) {unsupported} are recognized atmospheric "
                f"surface variables but are not yet supported as flux terms. "
                f"Supported: {sorted(_NAME_TO_ATOMIC_CONTRIBUTIONS)}"
            )


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig(CorrectorConfigABC):
    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    surface_energy_flux_correction: SurfaceEnergyFluxCorrectionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        state_copy = dict(state)
        if "masking" in state_copy:
            del state_copy["masking"]
        if "ocean_heat_content_correction" in state_copy and isinstance(
            state_copy["ocean_heat_content_correction"], bool
        ):
            if state_copy["ocean_heat_content_correction"]:
                state_copy["ocean_heat_content_correction"] = (
                    OceanHeatContentBudgetConfig(method="scaled_temperature")
                )
            else:
                state_copy["ocean_heat_content_correction"] = None
        if "sea_ice_fraction_correction" in state_copy:
            sif = state_copy["sea_ice_fraction_correction"]
            if isinstance(sif, dict) and "sea_ice_thickness_name" in sif:
                thickness_name = sif.pop("sea_ice_thickness_name")
                if thickness_name is not None:
                    sif.setdefault("zero_where_ice_free_names", []).append(
                        thickness_name
                    )
        return state_copy

    def get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "OceanCorrector":
        return OceanCorrector(
            self,
            dataset_info.gridded_operations,
            dataset_info.ocean_vertical_coordinate,
            dataset_info.timestep,
        )


class OceanCorrector(CorrectorABC):
    def __init__(
        self,
        config: OceanCorrectorConfig,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasOceanDepthIntegral | None,
        timestep: datetime.timedelta,
    ):
        self._config = config
        self._gridded_operations = gridded_operations
        self._vertical_coordinate = vertical_coordinate
        self._timestep = timestep

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict:
        if len(self._config.force_positive_names) > 0:
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        if self._config.sea_ice_fraction_correction is not None:
            gen_data = self._config.sea_ice_fraction_correction(gen_data, input_data)
        if self._config.surface_energy_flux_correction is not None:
            gen_data = _correct_hfds(
                input_data,
                gen_data,
                forcing_data,
                method=self._config.surface_energy_flux_correction.method,
                names=self._config.surface_energy_flux_correction.names,
            )
        if self._config.ocean_heat_content_correction is not None:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "Ocean heat content correction is turned on, but no vertical "
                    "coordinate is available."
                )
            gen_data = _force_conserve_ocean_heat_content(
                input_data,
                gen_data,
                forcing_data,
                self._gridded_operations.area_weighted_mean,
                self._vertical_coordinate,
                self._timestep.total_seconds(),
                self._config.ocean_heat_content_correction.method,
                self._config.ocean_heat_content_correction.constant_unaccounted_heating,
            )
        return dict(gen_data)


def _compute_ocean_net_surface_energy_flux(
    forcing_data: TensorMapping,
    sst: torch.Tensor,
    names: Literal["default"] | list[str] = "default",
) -> torch.Tensor:
    """Compute the net surface energy flux into the ocean from atmospheric
    forcing variables and the sea surface temperature.

    With ``names="default"``, the default combination is computed: the
    atmosphere net surface energy flux plus SST-dependent mass-heat transport
    by precipitation, frozen precipitation, and evaporation.

    With a list of ``names``, each name is decomposed into atomic contributions
    via ``_NAME_TO_ATOMIC_CONTRIBUTIONS`` and the union of contributions across
    all names is summed.
    """
    atmos = AtmosphereData(forcing_data)
    if names == "default":
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
    atomic_ids: set[str] = set().union(
        *(_NAME_TO_ATOMIC_CONTRIBUTIONS[n] for n in names)
    )
    total = torch.zeros_like(sst)
    # Sort for reproducible summation order across runs (set iteration order is
    # otherwise hash-randomized).
    for atomic_id in sorted(atomic_ids):
        try:
            total = total + _ATOMIC_CONTRIBUTIONS[atomic_id](atmos, sst)
        except KeyError as e:
            raise RuntimeError(
                f"Cannot compute surface energy flux contribution {atomic_id!r} "
                f"required by names={names}: forcing data is missing variable "
                f"{e.args[0]!r}."
            ) from e
    return total


def _correct_hfds(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    method: Literal["residual_prediction", "prescribed"],
    names: Literal["default"] | list[str] = "default",
) -> TensorDict:
    """Apply surface energy flux correction to the generated hfds.

    The ocean_fraction naturally zeroes the correction on land and reduces
    it under sea ice.

    Methods:
        residual_prediction: gen_hfds + ocean_fraction * net_flux
        prescribed: net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    """
    input = OceanData(input_data)
    forcing = OceanData(forcing_data)
    ocean_fraction = input.ocean_fraction
    net_flux = _compute_ocean_net_surface_energy_flux(
        forcing_data, input.sea_surface_temperature, names=names
    )
    out = dict(gen_data)
    if "hfds" in gen_data:
        hfds_name = "hfds"
    else:
        hfds_name = "hfds_total_area"
        net_flux = net_flux * forcing.sea_surface_fraction
    gen_hfds = gen_data[hfds_name]
    if method == "residual_prediction":
        out[hfds_name] = net_flux * ocean_fraction + gen_hfds
    elif method == "prescribed":
        out[hfds_name] = net_flux * ocean_fraction + gen_hfds * (1 - ocean_fraction)
    else:
        raise NotImplementedError(
            f"Method {method!r} not implemented for surface energy flux correction"
        )
    return out


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool, name: str | None = None
    ) -> torch.Tensor: ...


def _force_conserve_ocean_heat_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
    method: Literal["scaled_temperature"] = "scaled_temperature",
    unaccounted_heating: float = 0.0,
) -> TensorDict:
    if method != "scaled_temperature":
        raise NotImplementedError(
            f"Method {method!r} not implemented for ocean heat content conservation"
        )
    if "hfds" in gen_data and "hfds" in forcing_data:
        raise ValueError(
            "Net downward surface heat flux cannot be present in both gen_data and "
            "forcing_data."
        )
    input = OceanData(input_data, vertical_coordinate)
    if input.ocean_heat_content is None:
        raise ValueError(
            "ocean_heat_content is required to force ocean heat content conservation"
        )
    gen = OceanData(gen_data, vertical_coordinate)
    forcing = OceanData(forcing_data)
    global_gen_ocean_heat_content = area_weighted_mean(
        gen.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    global_input_ocean_heat_content = area_weighted_mean(
        input.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    try:
        # First priority: pre-weighted heat flux in gen_data
        net_energy_flux_into_ocean = (
            gen.net_downward_surface_heat_flux_total_area
            + forcing.geothermal_heat_flux * forcing.sea_surface_fraction
        )
    except KeyError:
        try:
            # Second priority: standard heat flux in gen_data
            net_energy_flux_into_ocean = (
                gen.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
        except KeyError:
            # Third priority: standard heat flux in input_data
            net_energy_flux_into_ocean = (
                input.net_downward_surface_heat_flux + forcing.geothermal_heat_flux
            ) * forcing.sea_surface_fraction
    energy_flux_global_mean = area_weighted_mean(
        net_energy_flux_into_ocean,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_change_ocean_heat_content = (
        energy_flux_global_mean + unaccounted_heating
    ) * timestep_seconds
    heat_content_correction_ratio = (
        global_input_ocean_heat_content + expected_change_ocean_heat_content
    ) / global_gen_ocean_heat_content
    # apply same temperature correction to all vertical layers
    n_levels = gen.sea_water_potential_temperature.shape[-1]
    for k in range(n_levels):
        name = f"thetao_{k}"
        gen.data[name] = gen.data[name] * heat_content_correction_ratio
    if "sst" in gen.data:
        gen.data["sst"] = (  # assuming sst in Kelvin
            gen.data["sst"] - FREEZING_TEMPERATURE_KELVIN
        ) * heat_content_correction_ratio + FREEZING_TEMPERATURE_KELVIN
    return gen.data
