import dataclasses
import datetime
from typing import Any, List, Mapping, Optional

import dacite

from fme.core.corrector.corrector import force_positive
from fme.core.corrector.registry import CorrectorABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.masking import StaticMaskingConfig
from fme.core.ocean_data import HasOceanIntegral
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorMapping


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig:
    masking: Optional[StaticMaskingConfig] = None
    force_positive_names: List[str] = dataclasses.field(default_factory=list)

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
        vertical_coordinate: Optional[HasOceanIntegral],
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

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorMapping:
        if self._masking is not None:
            gen_data = self._masking(gen_data)
        if len(self._config.force_positive_names) > 0:
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        return gen_data
