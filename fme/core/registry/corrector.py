import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar  # noqa: UP035

from fme.core.corrector.registry import CorrectorABC, CorrectorConfigABC
from fme.core.dataset_info import DatasetInfo

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
        super().__post_init__()
        if self.corrector_disabled_epochs != 0:
            raise ValueError(
                "corrector_disabled_epochs must be set on the wrapped corrector "
                "config (inside `config:`), not on the CorrectorSelector."
            )
        self._corrector_config_instance = self.registry.get(self.type, self.config)

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)

    @classmethod
    def get_available_types(cls) -> set[str]:
        """This class method is used to expose all available types of Correctors."""
        return set(cls.registry._types.keys())

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> CorrectorABC:
        # The wrapped config's get_corrector applies its own
        # corrector_disabled_epochs; the selector never schedules (guarded in
        # __post_init__), so no double-wrapping is possible.
        return self._corrector_config_instance.get_corrector(dataset_info)
