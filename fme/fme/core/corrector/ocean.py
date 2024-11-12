import dataclasses
import datetime
from types import MappingProxyType
from typing import Any, List, Mapping, Optional

import dacite

from fme.core.corrector import CorrectorABC, CorrectorConfigProtocol, force_positive
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.gridded_ops import GriddedOperations
from fme.core.masking import MaskingConfig
from fme.core.registry.corrector import CorrectorSelector
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorMapping

OCEAN_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "surface_height": ["zos"],
        "salinity": ["so_"],
        "potential_temperature": ["thetao_"],
        "zonal_velocity": ["uo_"],
        "meridional_velocity": ["vo_"],
    }
)


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig(CorrectorConfigProtocol):
    masking: Optional[MaskingConfig] = None
    force_positive_names: List[str] = dataclasses.field(default_factory=list)

    def build(
        self,
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ):
        return OceanCorrector(
            config=self,
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
            timestep=timestep,
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
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ):
        self._config = config
        self._gridded_operations = gridded_operations
        self._sigma_coordinates = sigma_coordinates
        self._timestep = timestep

        if config.masking is not None:
            self._masking = config.masking.build()
        else:
            self._masking = None
        self._stacker = Stacker(OCEAN_FIELD_NAME_PREFIXES)

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
    ) -> TensorMapping:
        if self._masking is not None:
            gen_data = self._masking(self._stacker, gen_data, input_data)
        if len(self._config.force_positive_names) > 0:
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        return gen_data
