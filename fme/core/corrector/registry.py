import abc
import datetime
from typing import Any

from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping


class CorrectorConfigABC(abc.ABC):
    @abc.abstractmethod
    def get_corrector(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: Any | None,
        timestep: datetime.timedelta,
    ) -> "CorrectorABC": ...


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict: ...
