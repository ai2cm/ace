import abc
from collections.abc import Mapping
from typing import Any, Self

import dacite

from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorDict, TensorMapping


class CorrectorConfigABC(abc.ABC):
    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        return dacite.from_dict(cls, state, config=dacite.Config(strict=True))

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
