import dataclasses
import datetime
from typing import Any, List, Mapping, Optional

import dacite
import torch

from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.utils import force_positive
from fme.core.gridded_ops import GriddedOperations
from fme.core.masking import StaticMaskingConfig
from fme.core.ocean_data import HasOceanDepthIntegral
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
    sea_ice_thickness_name: Optional[str] = None
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
    force_positive_names: List[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: Optional[SeaIceFractionConfig] = None
    # NOTE: OceanCorrector.masking is deprecated and kept for backwards
    # compatibility with legacy SingleModuleStepperConfig checkpoints. Please
    # use SingleModuleStepConfig.input_masking instead.
    masking: Optional[StaticMaskingConfig] = None

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
        vertical_coordinate: Optional[HasOceanDepthIntegral],
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
        if self._masking is not None:
            # NOTE: masking should be applied last to avoid overwriting
            gen_data = self._masking(gen_data)
        return dict(gen_data)
