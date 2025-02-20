import dataclasses
from typing import Any, List, Mapping, Optional

import dacite

from fme.core.corrector.corrector import force_positive
from fme.core.corrector.registry import CorrectorABC
from fme.core.masking import MaskingConfig
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorMapping


@CorrectorSelector.register("ocean_corrector")
@dataclasses.dataclass
class OceanCorrectorConfig:
    masking: Optional[MaskingConfig] = None
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
    ):
        self._config = config

        if config.masking is not None:
            self._masking = config.masking.build()
        else:
            self._masking = None

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorMapping:
        if self._masking is not None:
            gen_data = self._masking(gen_data, forcing_data)
        if len(self._config.force_positive_names) > 0:
            gen_data = force_positive(gen_data, self._config.force_positive_names)
        return gen_data
