import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Protocol

import dacite
import torch

from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.utils import force_positive
from fme.core.gridded_ops import GriddedOperations
from fme.core.masking import StaticMaskingConfig
from fme.core.ocean_data import HasOceanDepthIntegral, OceanData
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class SeaIceFractionConfig:
    """Correct predicted sea_ice_fraction to ensure it is always in 0-1, and
    land_fraction + sea_ice_fraction + ocean_fraction = 1. After
    sea_ice_fraction is corrected, if the sea_ice_thickness_name is provided
    it will be set to 0 everywhere the sea_ice_fraction is 0.
    """

    sea_ice_fraction_name: str
    land_fraction_name: str
    sea_ice_thickness_name: str | None = None
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
        if self.sea_ice_thickness_name:
            thickness = gen_data[self.sea_ice_thickness_name]
            thickness = thickness * (out[self.sea_ice_fraction_name] > 0.0)
            out[self.sea_ice_thickness_name] = thickness
        return out


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig:
    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: SeaIceFractionConfig | None = None
    # NOTE: OceanCorrector.masking is deprecated and kept for backwards
    # compatibility with legacy SingleModuleStepperConfig checkpoints. Please
    # use SingleModuleStepConfig.input_masking instead.
    masking: StaticMaskingConfig | None = None
    ocean_heat_content_correction: bool = False

    def __post_init__(self):
        if self.masking is not None and self.masking.mask_value != 0:
            raise ValueError(
                "mask_value must be 0 for OceanCorrector, but got "
                f"{self.masking.mask_value}"
            )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "OceanCorrectorConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
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

        if config.masking is not None:
            if vertical_coordinate is None:
                raise ValueError(
                    "OceanCorrector.masking configured but DepthCoordinate missing."
                )
            self._masking = config.masking.build(vertical_coordinate)
        else:
            self._masking = None

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
        if self._config.ocean_heat_content_correction:
            if self._vertical_coordinate is None:
                raise ValueError(
                    "Ocean heat content correction is turned on, but no vertical "
                    "coordinate is available."
                )
            gen_data = _force_conserve_ocean_heat_content(
                input_data,
                gen_data,
                self._gridded_operations.area_weighted_sum,
                self._vertical_coordinate,
                self._timestep.total_seconds(),
            )
        if self._masking is not None:
            # NOTE: masking should be applied last to avoid overwriting
            gen_data = self._masking(gen_data)
        return dict(gen_data)


class AreaWeightedSum(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool, name: str | None = None
    ) -> torch.Tensor: ...


def _force_conserve_ocean_heat_content(
    input_data: TensorMapping,
    gen_data: TensorMapping,
    area_weighted_sum: AreaWeightedSum,
    vertical_coordinate: HasOceanDepthIntegral,
    timestep_seconds: float,
) -> TensorDict:
    input = OceanData(input_data, vertical_coordinate)
    if input.ocean_heat_content is None:
        raise ValueError(
            "ocean_heat_content is required to force ocean heat content conservation"
        )
    gen = OceanData(gen_data, vertical_coordinate)
    global_gen_ocean_heat_content = area_weighted_sum(
        gen.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    global_input_ocean_heat_content = area_weighted_sum(
        input.ocean_heat_content,
        keepdim=True,
        name="ocean_heat_content",
    )
    expected_change_ocean_heat_content = area_weighted_sum(
        (input.net_downward_surface_heat_flux + input.geothermal_heat_flux)
        * input.sea_surface_fraction
        * timestep_seconds,
        keepdim=True,
        name="ocean_heat_content",
    )
    heat_content_correction_ratio = (
        global_input_ocean_heat_content + expected_change_ocean_heat_content
    ) / (global_gen_ocean_heat_content)

    # apply same temperature correction to all vertical layers
    n_levels = gen.sea_water_potential_temperature.shape[-1]
    for k in range(n_levels):
        name = f"thetao_{k}"
        gen.data[name] = gen.data[name] * torch.nan_to_num(
            heat_content_correction_ratio, nan=1.0
        )

    return gen.data
