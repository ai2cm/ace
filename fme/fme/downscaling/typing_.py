import dataclasses
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class HighResLowResPair(Generic[T]):
    highres: T
    lowres: T
