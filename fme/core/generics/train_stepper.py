import abc
from typing import Any, Generic, TypeVar

from torch import nn

from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.training_history import TrainingJob
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
    def get_state(self) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def from_state(cls: type[SelfType], state: dict[str, Any]) -> SelfType:
        pass

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def predict_paired(self) -> PredictFunction[PS, FD, SD]:
        pass

    def set_eval(self) -> None:
        for module in self.modules:
            module.eval()

    def set_train(self) -> None:
        for module in self.modules:
            module.train()

    @abc.abstractmethod
    def update_training_history(self, training_job: TrainingJob) -> None:
        pass
