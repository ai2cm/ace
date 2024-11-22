import abc
from typing import Generic, Iterable, Protocol, Sized, TypeVar

T = TypeVar("T", covariant=True)


class DataLoader(Protocol, Generic[T], Sized, Iterable[T]):
    pass


PS = TypeVar("PS")
BD = TypeVar("BD")


class InferenceDataABC(abc.ABC, Generic[PS, BD]):
    @property
    @abc.abstractmethod
    def initial_condition(self) -> PS:
        ...

    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[BD]:
        ...
