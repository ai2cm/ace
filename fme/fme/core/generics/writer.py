import abc
from typing import Generic, TypeVar

PS = TypeVar("PS")
BD = TypeVar("BD")


class WriterABC(abc.ABC, Generic[PS, BD]):
    @abc.abstractmethod
    def save_initial_condition(
        self,
        ic_data: PS,
    ):
        ...

    @abc.abstractmethod
    def append_batch(
        self,
        batch: BD,
    ):
        """
        Append a batch of data to the file.

        Args:
            batch: Data to be written.
        """
        ...
