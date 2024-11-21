import abc
import datetime
from typing import Any, Mapping, Protocol

from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping


class CorrectorConfigProtocol(Protocol):
    def build(
        self,
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ) -> "CorrectorABC":
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CorrectorConfigProtocol":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        ...


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorMapping:
        ...
