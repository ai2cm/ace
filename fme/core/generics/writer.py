import abc
from typing import Any, Generic, TypeVar

PS = TypeVar("PS", contravariant=True)  # prognostic state
SD = TypeVar("SD", contravariant=True)  # stepped data


class WriterABC(abc.ABC, Generic[PS, SD]):
    @abc.abstractmethod
    def write(self, data: PS, filename: str):
        """Eagerly write data to a file at filename."""
        ...

    @abc.abstractmethod
    def write_stepper_state(self, data: PS, filename: str):
        """Write the prognostic state's ``StepperState`` to a restart sidecar.

        A no-op when the state carries no ``StepperState`` (e.g. an unseeded,
        no-corrector-state rollout), so the sidecar is written only when there
        is something to restore.
        """
        ...

    @abc.abstractmethod
    def append_batch(
        self,
        batch: SD,
    ):
        """
        Append a batch of data to the output file(s).

        Args:
            batch: Data to be written.
        """
        ...


class NullDataWriter(WriterABC[Any, Any]):
    """
    Null pattern for DataWriter, which does nothing.
    """

    def __init__(self):
        pass

    def append_batch(
        self,
        batch: Any,
    ):
        pass

    def flush(self):
        pass

    def write(self, data: Any, filename: str):
        pass

    def write_stepper_state(self, data: Any, filename: str):
        pass

    def finalize(self):
        pass
