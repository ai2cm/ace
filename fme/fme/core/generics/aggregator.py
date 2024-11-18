import abc
from typing import Any, Dict, Generic, List, TypeVar

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
    ) -> InferenceLogs:
        """
        Record a batch of data.

        Args:
            data: Batch of data.

        Returns:
            Logs for the batch.
        """
        pass

    @abc.abstractmethod
    def record_initial_condition(
        self,
        initial_condition: PS,
    ) -> InferenceLogs:
        """
        Record the initial condition.

        May only be recorded once, before any calls to record_batch.
        """
        pass

    @abc.abstractmethod
    def get_summary_logs(self) -> InferenceLog:
        pass
