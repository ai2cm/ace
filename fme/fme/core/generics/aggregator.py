import abc
from typing import Any, Callable, Dict, Generic, List, TypeVar

from fme.core.typing_ import TensorDict, TensorMapping

PS = TypeVar("PS", contravariant=True)  # prognostic state
BD = TypeVar("BD")  # batch data


class AggregatorABC(abc.ABC, Generic[BD]):
    @abc.abstractmethod
    def record_batch(self, batch: BD) -> None:
        pass

    @abc.abstractmethod
    def get_logs(self, label: str) -> Dict[str, float]:
        pass


InferenceLog = Dict[str, Any]
InferenceLogs = List[InferenceLog]


class InferenceAggregatorABC(abc.ABC, Generic[PS, BD]):
    @abc.abstractmethod
    def record_batch(
        self,
        data: BD,
        normalize: Callable[[TensorMapping], TensorDict],
    ) -> InferenceLogs:
        pass

    @abc.abstractmethod
    def record_initial_condition(
        self,
        initial_condition: PS,
        normalize: Callable[[TensorMapping], TensorDict],
    ) -> InferenceLogs:
        pass

    @abc.abstractmethod
    def get_summary_logs(self) -> InferenceLog:
        pass
