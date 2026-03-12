import dataclasses
import datetime
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of CorrectorSelector
from typing import Any, ClassVar, Type, TypeVar, cast  # noqa: UP035

from fme.core.corrector.registry import CorrectorABC, CorrectorConfigABC
from fme.core.gridded_ops import GriddedOperations

from .registry import Registry

T = TypeVar("T")


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
    registry: ClassVar[Registry] = Registry()

    def __post_init__(self):
        self._corrector_config_instance: CorrectorConfigABC = cast(
            CorrectorConfigABC, self.registry.get(self.type, self.config)
        )

    @property
    def config_instance(self) -> CorrectorConfigABC:
        return self._corrector_config_instance

    @classmethod
    def register(cls, type_name) -> Callable[[Type[T]], Type[T]]:  # noqa: UP006
        return cls.registry.register(type_name)

    @classmethod
    def get_available_types(cls) -> set[str]:
        """This class method is used to expose all available types of Correctors."""
        return set(cls.registry._types.keys())

    @property
    def input_names(self) -> list[str]:
        return self._corrector_config_instance.input_names

    @property
    def next_step_input_names(self) -> list[str]:
        return self._corrector_config_instance.next_step_input_names

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
