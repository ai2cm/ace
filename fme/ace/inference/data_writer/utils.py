from dataclasses import dataclass
from typing import Iterable, Optional, Set, TypeVar

T = TypeVar("T")


def get_all_names(
    *data_varnames: Iterable[T], allowlist: Optional[Iterable[T]] = None
) -> Set[T]:
    """
    Returns all variable names from lists of variable names, optionally
    filtering by an allowlist.
    """
    variables: Set[T] = set()
    for varnames in data_varnames:
        variables = variables.union(set(varnames))
    if allowlist is None:
        return variables
    else:
        return variables.intersection(set(allowlist))


@dataclass
class DimInfo:
    name: str
    index: int


DIM_INFO_LATLON = [
    DimInfo(name="lat", index=-2),
    DimInfo(name="lon", index=-1),
]

DIM_INFO_HEALPIX = [
    DimInfo(name="face", index=-3),
    DimInfo(name="height", index=-2),
    DimInfo(name="width", index=-1),
]
