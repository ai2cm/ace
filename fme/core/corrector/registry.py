import abc
from collections.abc import Mapping
from typing import Any, Self, final

import dacite

from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorDict, TensorMapping


class CorrectorConfigABC(abc.ABC):
    @classmethod
    @final
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(cls, state, config=dacite.Config(strict=True))

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        """
        This method is used to remove or transform any deprecated keys from the
        state dict before loading it into a CorrectorConfigABC instance. It is
        optional to implement this method on subclasses.
        """
        return dict(state)

    @abc.abstractmethod
    def get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "CorrectorABC": ...


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict: ...

    @abc.abstractmethod
    def to(self, device: str) -> "CorrectorABC": ...
