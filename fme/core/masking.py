import dataclasses
from typing import List, Optional

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
class StaticMaskingConfig:
    """
    Replace static masked regions with a fill value, with handling for both 2D and 3D
    variables.

    Parameters:
        variable_names_and_prefixes: Names (2D variables) and prefixes (3D variables)
            to mask, which are used to build a variable Stacker.
        mask_value: Value of the mask variable in masked regions. Either 0 or 1.
        fill_value: The constant fill value to use outside of masked regions.
    """

    variable_names_and_prefixes: List[str]
    mask_value: int
    fill_value: float = 0.0

    def __post_init__(self):
        if self.mask_value not in [0, 1]:
            raise ValueError(
                "mask_value must be either 0 or 1, but got " f"{self.mask_value}"
            )

    def build(
        self,
        mask_2d: Optional[torch.Tensor] = None,
        mask_3d: Optional[torch.Tensor] = None,
    ):
        """
        Build Masking from config and 2D/3D masks. At least one of mask_2d
        and mask_3d must be non-null.

        mask_2d: 2D mask with shape (n_lat, n_lon).
        mask_3d: 3D mask with shape (n_lat, n_lon, n_levels).

        """
        return StaticMasking(
            variable_names_and_prefixes=self.variable_names_and_prefixes,
            mask_value=self.mask_value,
            fill_value=self.fill_value,
            mask_2d=mask_2d,
            mask_3d=mask_3d,
        )


class StaticMasking:
    def __init__(
        self,
        variable_names_and_prefixes: List[str],
        mask_value: int,
        fill_value: float,
        mask_2d: Optional[torch.Tensor] = None,
        mask_3d: Optional[torch.Tensor] = None,
    ):
        if mask_2d is None and mask_3d is None:
            raise ValueError("Either mask_2d or mask_3d must be provided to Masking.")
        if mask_2d is not None and len(mask_2d.shape) != 2:
            raise ValueError(
                f"Expected mask_2d with 2 dimensions, but got shape: {mask_2d.shape}"
            )
        if mask_3d is not None and len(mask_3d.shape) != 3:
            raise ValueError(
                f"Expected mask_3d with 3 dimensions, but got shape: {mask_3d.shape}"
            )
        self._mask_value = mask_value
        self._fill_value = fill_value
        # add convenience 3rd dimension to
        self._mask_2d = None if mask_2d is None else mask_2d.unsqueeze(-1)
        self._mask_3d = mask_3d
        self._stacker = Stacker({name: [name] for name in variable_names_and_prefixes})

    def __call__(self, data: TensorMapping) -> TensorDict:
        """
        Apply masking to the data for standard names recognized by a stacker.

        Args:
            data: The data to mask.

        """
        data_: TensorDict = {**data}
        for name in self._stacker.standard_names:
            stacked = self._stacker(name, data)
            mask = None
            if stacked.size(-1) > 1:  # 3D masking
                mask = self._mask_3d
                error_msg = (
                    f"{name[:-1]} is a 3D variable with shape {stacked.shape} "
                    "but Masking wasn't initialized with a 3D mask."
                )
            else:  # 2D masking
                mask = self._mask_2d
                stacked = stacked
                error_msg = (
                    f"{name} is a 2D variable with shape {stacked.squeeze().shape} "
                    "but Masking wasn't initialized with a 2D mask."
                )
            if mask is None:
                raise RuntimeError(error_msg)
            fill = torch.full_like(stacked, self._fill_value)
            masked = replace_on_mask(
                original=stacked,
                replacement=fill,
                mask=mask,
                mask_value=self._mask_value,
            )
            level_names = self._stacker.get_all_level_names(name, data)
            data_.update(unstack(masked, level_names, dim=-1))
        return data_
