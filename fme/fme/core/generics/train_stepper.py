import abc
from typing import Any, Dict, Generic, Type, TypeVar

from torch import nn

from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.typing_ import TensorDict

TO = TypeVar("TO", bound="TrainOutputABC")  # train output


class TrainOutputABC(abc.ABC):
    @abc.abstractmethod
    def get_metrics(self) -> TensorDict:
        pass


PS = TypeVar("PS")  # prognostic state
BD = TypeVar("BD")  # batch data
FD = TypeVar("FD")  # forcing data
SD = TypeVar("SD")  # stepped data


class TrainStepperABC(abc.ABC, Generic[PS, BD, FD, SD, TO]):
    SelfType = TypeVar("SelfType", bound="TrainStepperABC")

    @abc.abstractmethod
    def train_on_batch(
        self,
        data: BD,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> TO:
        pass

    @property
    @abc.abstractmethod
    def modules(self) -> nn.ModuleList:
        pass

    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def from_state(cls: Type[SelfType], state: Dict[str, Any]) -> SelfType:
        pass

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def predict_paired(self) -> PredictFunction[PS, FD, SD]:
        pass
