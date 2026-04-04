import dataclasses

from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class CoupledNames:
    ocean: list[str]
    atmosphere: list[str]


@dataclasses.dataclass
class CoupledTensorMapping:
    ocean: TensorMapping
    atmosphere: TensorMapping
