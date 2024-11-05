import abc
from typing import Callable, Dict, Generic, List, TypeVar

from fme.core.typing_ import TensorDict, TensorMapping

PS = TypeVar("PS")  # prognostic state
BD = TypeVar("BD")  # batch data


class AggregatorABC(abc.ABC, Generic[BD]):
    @abc.abstractmethod
    def record_batch(self, batch: BD) -> None:
        pass

    @abc.abstractmethod
    def get_logs(self, label: str) -> Dict[str, float]:
        pass


class InferenceAggregatorABC(abc.ABC, Generic[PS, BD]):
    @abc.abstractmethod
    def record_batch(
        self,
        data: BD,
        normalize: Callable[[TensorMapping], TensorDict],
        i_time_start: int,
    ) -> None:
        pass

    @abc.abstractmethod
    def record_initial_condition(
        self,
        initial_condition: PS,
        normalize: Callable[[TensorMapping], TensorDict],
    ) -> None:
        pass

    @abc.abstractmethod
    def get_logs(self, label: str) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def get_inference_logs_slice(
        self, label: str, step_slice: slice
    ) -> List[Dict[str, float | int]]:
        pass

    @property
    @abc.abstractmethod
    def log_time_series(self) -> bool:
        pass
