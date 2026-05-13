import dataclasses

from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class CoupledNames:
    ocean: list[str]
    atmosphere: list[str]
    ice: list[str]


@dataclasses.dataclass
class CoupledTensorMapping:
    ocean: TensorMapping
    atmosphere: TensorMapping
    ice: TensorMapping


@dataclasses.dataclass
class CoupledOptionalInt:
    ocean: int | None
    atmosphere: int | None
    ice: int | None
