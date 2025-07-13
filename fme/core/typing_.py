import dataclasses
from collections.abc import Mapping
from typing import NewType

import torch

TensorMapping = Mapping[str, torch.Tensor]
TensorDict = dict[str, torch.Tensor]
EnsembleTensorDict = NewType("EnsembleTensorDict", TensorDict)
EnsembleTensorDict.__doc__ = """
A dictionary of tensors with an explicit ensemble (sample) dimension, where
ensemble members represent multiple predictions for the same initial condition.

The ensemble dimension is the second dimension of the tensors,
while the batch dimension is the first.
"""


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

    start: int | None = None
    stop: int | None = None
    step: int | None = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def contains(self, value: int) -> bool:
        start = self.start if self.start is not None else 0
        stop = self.stop if self.stop is not None else float("inf")
        step = self.step if self.step is not None else 1
        return start <= value < stop and (value - start) % step == 0
