import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, ClassVar

from fme.core.corrector.registry import CorrectorABC, CorrectorConfigABC
from fme.core.gridded_ops import GriddedOperations

from .registry import Registry


@dataclasses.dataclass
class CorrectorSelector(CorrectorConfigABC):
    """
    A dataclass containing all the information needed to build a
    CorrectorConfigABC, including the type of the CorrectorConfigABC and the
    data needed to build it.

    This is helpful as CorrectorSelector can be serialized and deserialized
    without any additional information, whereas to load a CorrectorConfigABC you
    would need to know the type of the CorrectorConfigABC being loaded.

    It is also convenient because CorrectorSelector is a single class that can
    be used to represent any CorrectorConfigABC, whereas CorrectorConfigABC is
    an ABC that can be implemented by many different classes.

    Parameters:
        type: the type of the CorrectorConfigABC
        config: data for a CorrectorConfigABC instance of the indicated type

    """

    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry[CorrectorConfigABC]] = Registry[CorrectorConfigABC]()

    def __post_init__(self):
        self._corrector_config_instance = self.registry.get(self.type, self.config)

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)

    @classmethod
    def get_available_types(cls) -> set[str]:
        """This class method is used to expose all available types of Correctors."""
        return set(cls.registry._types.keys())

    def get_corrector(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: Any | None,
        timestep: datetime.timedelta,
    ) -> CorrectorABC:
        return self._corrector_config_instance.get_corrector(
            gridded_operations,
            vertical_coordinate,
            timestep,
        )
