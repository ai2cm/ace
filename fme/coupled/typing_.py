import dataclasses

from fme.core.typing_ import TensorDict


@dataclasses.dataclass
class CoupledTensorDict:
    ocean: TensorDict
    atmosphere: TensorDict
