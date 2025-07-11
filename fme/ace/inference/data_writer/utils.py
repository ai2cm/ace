from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def get_all_names(
    *data_varnames: Iterable[T], allowlist: Iterable[T] | None = None
) -> set[T]:
    """
    Returns all variable names from lists of variable names, optionally
    filtering by an allowlist.
    """
    variables: set[T] = set()
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
