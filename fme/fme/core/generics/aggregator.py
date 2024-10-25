import abc
from typing import Callable, Dict, Generic, List, TypeVar

from fme.core.typing_ import TensorDict, TensorMapping

T = TypeVar("T")


class AggregatorABC(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def record_batch(self, batch: T) -> None:
        pass

    @abc.abstractmethod
    def get_logs(self, label: str) -> Dict[str, float]:
        pass


class InferenceEvaluatorAggregatorABC(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def record_batch(
        self,
        prediction: T,
        target: T,
        normalize: Callable[[TensorMapping], TensorDict],
        i_time_start: int,
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


class InferenceAggregatorABC(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def record_batch(
        self,
        prediction: T,
        normalize: Callable[[TensorMapping], TensorDict],
        i_time_start: int,
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
