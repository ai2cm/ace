import abc
from typing import Generic, TypeVar

PS = TypeVar("PS", contravariant=True)  # prognostic state
SD = TypeVar("SD", contravariant=True)  # stepped data


class WriterABC(abc.ABC, Generic[PS, SD]):
    @abc.abstractmethod
    def write(self, data: PS, filename: str):
        """Eagerly write data to a file at filename."""
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
