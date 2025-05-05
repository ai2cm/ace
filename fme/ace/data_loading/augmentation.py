import abc
import dataclasses
import re

import torch

from fme.ace.data_loading.batch_data import BatchData


@dataclasses.dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.

    Attributes:
        rotate_probability: The probability of rotating the sphere by 180 degrees,
            as a value between 0.0 and 1.0.
        additional_directional_names: Names of variables whose sign is flipped when
            the poles are reversed. By default this includes known directional
            names as stored in RotateModifier.FLIP_NAMES.
    """

    rotate_probability: float = 0.0
    additional_directional_names: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.rotate_probability <= 1.0:
            raise ValueError(
                "rotate_probability must be between 0.0 and 1.0, "
                f"got {self.rotate_probability}"
            )

    def build_modifier(self) -> "BatchModifierABC":
        if self.rotate_probability > 0.0:
            return RotateModifier(
                self.rotate_probability, self.additional_directional_names
            )
        return NullModifier()


class BatchModifierABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch: BatchData) -> BatchData: ...


class RotateModifier(BatchModifierABC):
    """
    Modifier that rotates the sphere by 180 degrees so that the poles swap
    places. This is the same as flipping both zonal and meridional axes.

    Also flips the sign of horizontal directional variables such as horizontal
    winds in specific directions, so their new values reflect the rotated axes.
    The names of such variables are stored in the `FLIP_NAMES` class variable.
    Variables not included in this list are not flipped.

    Specifically, the regex pattern r'{name}(_?[0-9]+m?)?$' is used to match the
    names of variables whose sign is flipped when the poles are reversed, for
    each name in `FLIP_NAMES`. This will match both names that end with something
    like "_0", "_1", etc. or something like "10m" or "2m".

    Note that seasons are handled by the fact that solar insolation is a data
    variable, but time is not modified. This means monthly or seasonal averages
    using this data will be affected by the rotation.
    """

    # names of variables whose sign is flipped when the poles are reversed
    FLIP_NAMES = [
        "eastward_wind",
        "northward_wind",
        "UGRD",
        "VGRD",
        "U",
        "V",
    ]

    def __init__(
        self,
        rotate_probability: float,
        additional_directional_names: list[str],
    ):
        self.rotate_probability = rotate_probability
        self.additional_directional_names = additional_directional_names
        self._pattern = re.compile(
            r"({})(_?[0-9]+m?)?$".format(
                "|".join(self.FLIP_NAMES + self.additional_directional_names)
            )
        )

    def __call__(self, batch: BatchData) -> BatchData:
        if batch.horizontal_dims != ["lat", "lon"]:
            raise NotImplementedError(
                "Horizontal dimensions must be lat and lon to rotate the sphere, got "
                f"{batch.horizontal_dims}"
            )
        example_value = next(iter(batch.data.values()))
        apply = torch.rand(example_value.shape[0]) < self.rotate_probability
        while len(apply.shape) < len(example_value.shape):
            apply = apply.unsqueeze(-1)
        new_data = {}
        for name, value in batch.data.items():
            new_value = torch.flip(value, dims=[-2, -1])
            if self._pattern.match(name):
                new_value = -1 * new_value
            new_data[name] = torch.where(apply, new_value, value)
        return BatchData(new_data, batch.time, batch.horizontal_dims)


class NullModifier(BatchModifierABC):
    def __call__(self, batch: BatchData) -> BatchData:
        return batch
