import dataclasses
from typing import Dict, Mapping, Optional

import torch

TensorMapping = Mapping[str, torch.Tensor]
TensorDict = Dict[str, torch.Tensor]


@dataclasses.dataclass
class Slice:
    """
    Configuration of a python `slice` built-in.

    Required because `slice` cannot be initialized directly by dacite.

    Parameters:
        start: Start index of the slice.
        stop: Stop index of the slice.
        step: Step of the slice.
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)
