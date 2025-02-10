import dataclasses
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class FineResCoarseResPair(Generic[T]):
    fine: T
    coarse: T
