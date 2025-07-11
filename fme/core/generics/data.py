import abc
from collections.abc import Iterable, Sized
from typing import Generic, Protocol, TypeVar

T = TypeVar("T", covariant=True)


class DataLoader(Protocol, Generic[T], Sized, Iterable[T]):
    pass


PS = TypeVar("PS")  # prognostic state
FD = TypeVar("FD", covariant=True)  # forcing data


class InferenceDataABC(abc.ABC, Generic[PS, FD]):
    @property
    @abc.abstractmethod
    def initial_condition(self) -> PS: ...

    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[FD]: ...


class SimpleInferenceData(InferenceDataABC[PS, FD]):
    def __init__(
        self,
        initial_condition: PS,
        loader: DataLoader[FD],
    ):
        self._initial_condition = initial_condition
        self._loader = loader

    @property
    def initial_condition(self) -> PS:
        return self._initial_condition

    @property
    def loader(self) -> DataLoader[FD]:
        return self._loader


class GriddedDataABC(abc.ABC, Generic[T]):
    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[T]: ...

    @property
    @abc.abstractmethod
    def n_samples(self) -> int: ...

    @property
    @abc.abstractmethod
    def n_batches(self) -> int: ...

    @property
    @abc.abstractmethod
    def batch_size(self) -> int: ...

    @abc.abstractmethod
    def set_epoch(self, epoch: int): ...

    @abc.abstractmethod
    def log_info(self, name: str):
        """
        Report information about the data using logging.info.
        """
        ...
