import dataclasses

from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class CoupledTensorMapping:
    ocean: TensorMapping
    atmosphere: TensorMapping
