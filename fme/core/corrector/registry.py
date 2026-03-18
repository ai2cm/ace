import abc

from fme.core.dataset_info import DatasetInfo
from fme.core.typing_ import TensorDict, TensorMapping


class CorrectorConfigABC(abc.ABC):
    @abc.abstractmethod
    def get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "CorrectorABC": ...

    @property
    @abc.abstractmethod
    def input_names(self) -> list[str]:
        """Names of additional variables the corrector requires as inputs."""
        ...

    @property
    @abc.abstractmethod
    def next_step_input_names(self) -> list[str]:
        """Names of additional variables the corrector requires in
        next_step_input_data.
        """
        ...


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict: ...
