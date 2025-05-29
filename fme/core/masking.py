import collections
import dataclasses
import re
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

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


@runtime_checkable
class HasGetMaskTensorFor(Protocol):
    def build_output_masker(self) -> Callable[[TensorMapping], TensorDict]: ...

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        """Get the mask for a specific variable name."""
        ...

    def to(self, device: str) -> "HasGetMaskTensorFor": ...


@dataclasses.dataclass
class StaticMaskingConfig:
    """
    Replace static masked regions with a fill value.

    Parameters:
        mask_value: Value of the mask variable in masked regions. Either 0 or 1.
        fill_value: A float fill value to use outside of masked regions. Can also be
            "mean", in which case the normalizer means are used as channel-specific
            fill values.
        exclude_names_and_prefixes: Names (2D variables) and prefixes (3D variables)
            to exclude when applying the mask.

    """

    mask_value: int
    fill_value: Literal["mean"] | float = 0.0
    exclude_names_and_prefixes: list[str] | None = None

    def __post_init__(self):
        if self.mask_value not in [0, 1]:
            raise ValueError(
                f"mask_value must be either 0 or 1, but got {self.mask_value}"
            )

    def build(self, mask: HasGetMaskTensorFor, means: TensorMapping | None = None):
        """
        Build StaticMasking.

        """
        if isinstance(self.fill_value, float):
            return StaticMasking(
                mask_value=self.mask_value,
                fill_value=collections.defaultdict(
                    lambda: torch.as_tensor(self.fill_value)
                ),
                mask=mask,
                exclude_names_and_prefixes=self.exclude_names_and_prefixes,
            )
        if means is None:
            raise ValueError(
                "fill_values mapping required by build unless configured "
                "fill_value is a float."
            )
        return StaticMasking(
            mask_value=self.mask_value,
            fill_value=means,
            mask=mask,
            exclude_names_and_prefixes=self.exclude_names_and_prefixes,
        )


class StaticMasking:
    def __init__(
        self,
        mask_value: int,
        fill_value: float | TensorMapping,
        mask: HasGetMaskTensorFor,
        exclude_names_and_prefixes: list[str] | None = None,
    ):
        if isinstance(fill_value, float):
            fill_mapping: TensorMapping = collections.defaultdict(
                lambda: torch.as_tensor(fill_value)
            )
        else:
            fill_mapping = fill_value
        self._fill_mapping = fill_mapping
        self._mask_value = mask_value
        self._mask = mask
        self._exclude_regex = self._build_regex(exclude_names_and_prefixes)

    def _build_regex(self, names_and_prefixes: list[str] | None) -> str | None:
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

    def _masks(self, name: str):
        exclude = self._exclude_regex and re.match(self._exclude_regex, name)
        return not exclude

    def __call__(self, data: TensorMapping) -> TensorDict:
        """
        Apply masking to the data for standard names recognized by a stacker.

        Args:
            data: The data to mask.

        """
        data_: TensorDict = {**data}
        for name, tensor in data_.items():
            if not self._masks(name):
                continue
            mask = self._mask.get_mask_tensor_for(name)
            if mask is None:
                continue
            try:
                fill_value = self._fill_mapping[name]
            except KeyError as err:
                raise KeyError(
                    "StaticMasking was initialized with a fill_value mapping "
                    f"but the mapping is missing key '{name}'."
                ) from err
            fill = torch.full_like(tensor, fill_value)
            mask = mask.expand(fill.shape)
            masked = replace_on_mask(
                original=tensor,
                replacement=fill,
                mask=mask,
                mask_value=self._mask_value,
            )
            data_[name] = masked
        return data_


class NullMasking:
    def __call__(self, data: TensorMapping) -> TensorDict:
        return dict(data)
