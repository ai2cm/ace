import abc
from typing import Generic, TypeVar

T = TypeVar("T")


class PrognosticStateABC(abc.ABC, Generic[T]):
    SelfType = TypeVar("SelfType", bound="PrognosticStateABC")

    @abc.abstractmethod
    def as_state(self) -> T:
        ...
