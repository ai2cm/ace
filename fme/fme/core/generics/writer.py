import abc
from typing import Generic, TypeVar

PS = TypeVar("PS", contravariant=True)  # prognostic state
SD = TypeVar("SD", contravariant=True)  # stepped data


class WriterABC(abc.ABC, Generic[PS, SD]):
    @abc.abstractmethod
    def save_initial_condition(
        self,
        ic_data: PS,
    ):
        ...

    @abc.abstractmethod
    def append_batch(
        self,
        batch: SD,
    ):
        """
        Append a batch of data to the file.

        Args:
            batch: Data to be written.
        """
        ...
