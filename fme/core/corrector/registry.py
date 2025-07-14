import abc

from fme.core.typing_ import TensorDict, TensorMapping


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
    ) -> TensorDict: ...
