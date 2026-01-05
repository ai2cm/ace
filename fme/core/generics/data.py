import abc
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import Generic, Protocol, TypeVar

_T = TypeVar("_T", covariant=True)


class DataLoader(Protocol, Generic[_T], Sized, Iterable[_T]):
    pass


_U = TypeVar("_U")


class SizedMap(Generic[_T, _U], Sized, Iterable[_U]):
    def __init__(self, func: Callable[[_T], _U], iterable: DataLoader[_T]):
        self._func = func
        self._iterable = iterable

    def __len__(self) -> int:
        return len(self._iterable)

    def __iter__(self) -> Iterator[_U]:
        return map(self._func, self._iterable)


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


BD = TypeVar("BD", covariant=True)  # batch data


class GriddedDataABC(abc.ABC, Generic[BD]):
    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[BD]: ...

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
    def alternate_shuffle(self):
        """
        Change the random shuffle of the data loader for the current epoch.
        """
        ...

    @abc.abstractmethod
    def subset_loader(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ) -> DataLoader[BD]:
        """
        Subset the loader to skip the first `start_batch` batches
        and stop at the `stop_batch` batch (exclusive).
        """
        ...

    @abc.abstractmethod
    def log_info(self, name: str):
        """
        Report information about the data using logging.info.
        """
        ...
