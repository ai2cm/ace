import dataclasses
import re
from typing import List, Optional, Protocol

import torch

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


class HasGetMaskTensorFor(Protocol):
    def get_mask_tensor_for(self, name: str) -> Optional[torch.Tensor]:
        """Get the mask for a specific variable name."""
        ...


@dataclasses.dataclass
class StaticMaskingConfig:
    """
    Replace static masked regions with a fill value.

    Parameters:
        variable_names_and_prefixes: Names (2D variables) and prefixes (3D variables)
            to mask, which are used to build a variable Stacker.
        mask_value: Value of the mask variable in masked regions. Either 0 or 1.
        fill_value: The constant fill value to use outside of masked regions.
    """

    mask_value: int
    fill_value: float = 0.0
    variable_names_and_prefixes: Optional[List[str]] = None

    def __post_init__(self):
        if self.mask_value not in [0, 1]:
            raise ValueError(
                "mask_value must be either 0 or 1, but got " f"{self.mask_value}"
            )

    def build(self, mask: HasGetMaskTensorFor):
        """
        Build StaticMasking.

        """
        return StaticMasking(
            mask_value=self.mask_value,
            fill_value=self.fill_value,
            mask=mask,
            variable_names_and_prefixes=self.variable_names_and_prefixes,
        )


class StaticMasking:
    def __init__(
        self,
        mask_value: int,
        fill_value: float,
        mask: HasGetMaskTensorFor,
        variable_names_and_prefixes: Optional[List[str]] = None,
    ):
        self._fill_value = fill_value
        self._mask_value = mask_value
        self._mask = mask
        self._include_regex = self._build_regex(variable_names_and_prefixes)

    def _build_regex(self, names_and_prefixes: Optional[List[str]]) -> Optional[str]:
        if names_and_prefixes:
            regex = []
            for name in names_and_prefixes:
                if name.endswith("_"):
                    regex.append(rf"^{name}\d+$")
                elif not re.match(r".+_\d+$", name):
                    regex.append(f"^{name}$")
                    regex.append(rf"^{name}_\d+$")
                else:
                    regex.append(rf"^{name}$")
            return r"|".join(regex)
        return None

    def _include(self, name: str):
        return self._include_regex is None or re.match(self._include_regex, name)

    def __call__(self, data: TensorMapping) -> TensorDict:
        """
        Apply masking to the data for standard names recognized by a stacker.

        Args:
            data: The data to mask.

        """
        data_: TensorDict = {**data}
        for name, tensor in data_.items():
            if not self._include(name):
                continue
            mask = self._mask.get_mask_tensor_for(name)
            if mask is None:
                continue
            fill = torch.full_like(tensor, self._fill_value)
            masked = replace_on_mask(
                original=tensor,
                replacement=fill,
                mask=mask,
                mask_value=self._mask_value,
            )
            data_[name] = masked
        return data_
