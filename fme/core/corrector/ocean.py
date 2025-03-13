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
    sea_ice_fraction_name: str
    land_fraction_name: str


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig:
    masking: Optional[StaticMaskingConfig] = None
    force_positive_names: List[str] = dataclasses.field(default_factory=list)
    sea_ice_fraction_correction: Optional[SeaIceFractionConfig] = None

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
            self._masking = config.masking.build(
                mask_2d=vertical_coordinate.get_mask_level(0),
                mask_3d=vertical_coordinate.get_mask(),
            )
        else:
            self._masking = None

    def _correct_sea_ice_fraction(
        self, gen_data: TensorMapping, input_data: TensorMapping
    ) -> TensorDict:
        """Correct predicted sea ice fraction to ensure \
            sea ice fraction is always 0-1, and \
            land fraction + sea ice fraction + ocean fraction = 1.
        """
        out = {**gen_data}
        sea_ice_config = self._config.sea_ice_fraction_correction
        if sea_ice_config is not None:
            out[sea_ice_config.sea_ice_fraction_name] = torch.clamp(
                out[sea_ice_config.sea_ice_fraction_name], min=0.0, max=1.0
            )
            negative_ocean_fraction = (
                1
                - out[sea_ice_config.sea_ice_fraction_name]
                - input_data[sea_ice_config.land_fraction_name]
            )
            negative_ocean_fraction = negative_ocean_fraction.clip(max=0)
            out[sea_ice_config.sea_ice_fraction_name] += negative_ocean_fraction
        return out

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict:
        if self._masking is not None:
            gen_data = self._masking(gen_data)
        if len(self._config.force_positive_names) > 0:
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        gen_data = self._correct_sea_ice_fraction(gen_data, input_data)
        return gen_data
