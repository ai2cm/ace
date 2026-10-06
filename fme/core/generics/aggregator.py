import abc
import dataclasses
from typing import Any, Generic, TypeVar

PS = TypeVar("PS", contravariant=True)  # prognostic state
T = TypeVar("T", contravariant=True)


@dataclasses.dataclass
class AggregatorSummary:
    """Summary returned by training/validation aggregators.

    Attributes:
        logs: Metrics dict suitable for wandb logging.
        loss: Scalar to minimize for best-validation checkpoint selection,
            or ``None`` if this aggregator does not contribute to checkpoint
            selection.
    """

    logs: dict[str, float]
    loss: float | None


class AggregatorABC(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def record_batch(self, batch: T) -> None:
        pass

    @abc.abstractmethod
    def get_summary(self, label: str) -> AggregatorSummary:
        pass

    @abc.abstractmethod
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass


InferenceLog = dict[str, Any]
InferenceLogs = list[InferenceLog]


@dataclasses.dataclass
class InferenceSummary:
    """Summary returned by inference aggregators.

    Attributes:
        logs: Metrics dict suitable for wandb logging.
        loss: Scalar to minimize for best-inference checkpoint selection,
            or ``None`` if this aggregator does not contribute to checkpoint
            selection.
    """

    logs: InferenceLog
    loss: float | None


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
    def get_summary(self) -> InferenceSummary:
        pass

    @abc.abstractmethod
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass
