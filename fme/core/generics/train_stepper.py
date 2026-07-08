import abc
from typing import Any, Generic, TypeVar

from torch import nn

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
        evaluate_all_steps: bool = False,
    ) -> TO:
        pass

    @abc.abstractmethod
    def predict_paired(
        self,
        initial_condition: PS,
        forcing: FD,
        compute_derived_variables: bool = False,
    ) -> tuple[SD, PS]:
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

    @abc.abstractmethod
    def set_eval(self) -> None:
        pass

    @abc.abstractmethod
    def set_train(self) -> None:
        pass

    def set_epoch(self, epoch: int) -> None:
        """Called by the trainer at the start of each training epoch.

        Default implementation is a no-op. Override to reset per-epoch
        in-module state (e.g. tracked running statistics that should only
        reflect the most recent epoch).
        """
        pass

    @abc.abstractmethod
    def update_training_history(self, training_job: TrainingJob) -> None:
        pass

    @abc.abstractmethod
    def seed_eval(self, seed: int) -> None:
        pass
