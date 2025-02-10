import dataclasses
from typing import Optional

import torch

from fme.core.stacker import Stacker, unstack
from fme.core.typing_ import TensorDict, TensorMapping


def replace_on_mask(
    original: torch.Tensor,
    replacement: torch.Tensor,
    mask: torch.Tensor,
    mask_value: int,
):
    """Replace original with replacement in masked regions.

    Args:
        original: The original data tensor.
        replacement: The replacement data tensor.
        mask: The mask tensor.
        mask_value: The value of the mask variable in the region to be replaced.
    """
    rounded_mask = torch.round(mask).to(int)
    return torch.where(
        condition=rounded_mask == mask_value,
        input=replacement,
        other=original,
    )


@dataclasses.dataclass
class MaskingConfig:
    """
    Configuration for applying masking to the generated output.

    Parameters:
        mask_name: The standard name of the mask. May be a prefix (for a 3D masking
            variable) or a full name (for a 2D masking variable).
        mask_value: Value of the mask variable in masked regions. Either 0 or 1.
        fill_value: The constant fill value to use outside of masked regions.
        surface_mask_name: (optional) The full name of the surface mask. Only required
            when mask_name is a prefix and separate 2D surface masking is desired.
    """

    mask_name: str
    mask_value: int
    fill_value: float = 0.0
    surface_mask_name: Optional[str] = None

    def __post_init__(self):
        if self.mask_value not in [0, 1]:
            raise ValueError(
                "mask_value must be either 0 or 1, but got " f"{self.mask_value}"
            )

    def build(self):
        return Masking(
            mask_name=self.mask_name,
            mask_value=self.mask_value,
            fill_value=self.fill_value,
            surface_mask_name=self.surface_mask_name,
        )


class Masking:
    """Replace masked regions with a fill value."""

    def __init__(
        self,
        mask_name: str,
        mask_value: int,
        fill_value: float,
        surface_mask_name: Optional[str] = None,
    ):
        self.mask_name = mask_name
        self.mask_value = mask_value
        self.fill_value = fill_value
        self.surface_mask_name = surface_mask_name
        mask_map = {self.mask_name: [self.mask_name]}
        if self.surface_mask_name is not None:
            mask_map[self.surface_mask_name] = [self.surface_mask_name]

        self.mask_stacker = Stacker(mask_map)

    def __call__(
        self,
        stacker: Stacker,
        data: TensorMapping,
        mask_data: TensorMapping,
    ) -> TensorDict:
        """
        Apply masking to the data for standard names recognized by a stacker.

        Args:
            stacker: A Stacker for variables to mask in data.
            data: The data to mask.
            mask_data: The mask data.

        """
        mask = self.mask_stacker(self.mask_name, mask_data)
        if self.surface_mask_name is not None:
            surface_mask = self.mask_stacker(self.surface_mask_name, mask_data)
        else:
            surface_mask = None
        data_: TensorDict = {**data}
        for name in stacker.standard_names:
            stacked = stacker(name, data)
            if stacked.size(-1) > 1:  # 3D masking
                mask_ = mask
            elif surface_mask is not None:  # 2D masking with surface mask
                mask_ = surface_mask
            elif mask.size(-1) == 1:  # 2D masking with mask
                mask_ = mask
            else:
                raise RuntimeError(
                    "Masking surface_mask_name is None but the input Stacker "
                    f"includes the 2D standard name {name}."
                )
            fill = torch.full_like(stacked, self.fill_value)
            masked = replace_on_mask(
                original=stacked,
                replacement=fill,
                mask=mask_,
                mask_value=self.mask_value,
            )
            level_names = stacker.get_all_level_names(name, data)
            data_.update(unstack(masked, level_names, dim=-1))
        return data_
