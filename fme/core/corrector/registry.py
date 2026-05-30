import abc
import dataclasses
from collections.abc import Mapping
from typing import Any, Self, final

import dacite

from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class CorrectionResult:
    """The result of applying a corrector to generated data.

    Attributes:
        corrected: The corrected data, containing all variables passed to the
            corrector (corrected or not).
        before: The pre-correction values of exactly the variables the corrector
            modified. Variables the corrector did not touch are absent.
    """

    corrected: TensorDict
    before: TensorDict


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
    ) -> CorrectionResult: ...
