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


def _shift_bound(
    value: int | None, shift: int, out_of_bounds_value: int | None
) -> int | None:
    """
    Shifts a bounding value of a slice relative to a starting index where
    positive shift will shift the bound left (decrease the index),
    and negative shift will shift the bound right (increase the index).
    When shifting left, if the shifted value is less than 0,
    it is considered out of bounds and replaced with `out_of_bounds_value`.
    If the value is None, it remains None.  Negative initial bound values
    are not supported.
    """
    if value is None:
        return None
    elif value < 0:
        raise ValueError("Negative slice bounds as an initial value are not supported")

    shifted = value - shift
    if shifted < 0:
        return out_of_bounds_value

    return shifted


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

    @classmethod
    def shift_left(cls, original: "Slice", start_index: int) -> "Slice":
        """
        Shift the slice relative to the start index of a group of data to
        capture requested correct quantities while still respecting batches.
        E.g., If slice is (0, 10, 1) and start_index is 5, the new slice
        would be (None, 5, 1).

        Raises:
            ValueError: If trying to  shift negative valued slice object,
            since that is not defined without knowing the total sequence
            length.
        """
        new_start = _shift_bound(original.start, start_index, None)
        new_stop = _shift_bound(original.stop, start_index, 0)
        return cls(start=new_start, stop=new_stop, step=original.step)
