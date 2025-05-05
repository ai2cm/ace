import abc
from typing import Any, Generic, TypeVar

PS = TypeVar("PS", contravariant=True)  # prognostic state
T = TypeVar("T", contravariant=True)


class AggregatorABC(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def record_batch(self, batch: T) -> None:
        pass

    @abc.abstractmethod
    def get_logs(self, label: str) -> dict[str, float]:
        pass

    @abc.abstractmethod
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass


InferenceLog = dict[str, Any]
InferenceLogs = list[InferenceLog]


class InferenceAggregatorABC(abc.ABC, Generic[PS, T]):
    @abc.abstractmethod
    def record_batch(
        self,
        data: T,
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

    @abc.abstractmethod
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass
