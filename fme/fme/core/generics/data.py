import abc
from typing import Generic, Iterable, Protocol, Sized, TypeVar

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
