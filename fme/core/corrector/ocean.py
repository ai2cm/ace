import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import dacite
import torch

from fme.core.constants import FREEZING_TEMPERATURE_KELVIN
from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.utils import force_positive
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


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig:
    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    ocean_heat_content_correction: OceanHeatContentBudgetConfig | None = None

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OceanCorrectorConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

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
        elif (
            "ocean_heat_content_correction" in state_copy
            and "method" in state_copy["ocean_heat_content_correction"]
            and state_copy["ocean_heat_content_correction"]["method"]
            == "constant_temperature"
        ):
            # FIXME: don't merge!
            state_copy["ocean_heat_content_correction"]["method"] = "scaled_temperature"
        if "sea_ice_fraction_correction" in state_copy:
            sif = state_copy["sea_ice_fraction_correction"]
            if isinstance(sif, dict) and "sea_ice_thickness_name" in sif:
                thickness_name = sif.pop("sea_ice_thickness_name")
                if thickness_name is not None:
                    sif.setdefault("zero_where_ice_free_names", []).append(
                        thickness_name
                    )
        return state_copy


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
