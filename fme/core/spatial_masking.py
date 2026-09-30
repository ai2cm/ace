import collections
import dataclasses
from typing import Literal, Protocol, runtime_checkable

import torch

from fme.core.name_and_prefix_matcher import NameAndPrefixMatcher
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
class HasGetSpatialMask(Protocol):
    def build_output_spatial_masker(self) -> "SpatialMasking": ...

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        """Get the mask for a specific variable name."""
        ...

    def to(self, device: str) -> "HasGetSpatialMask": ...


@dataclasses.dataclass
class StaticSpatialMaskingConfig:
    """
    Replace static spatially masked regions with a fill value.

    Parameters:
        mask_value: Value of the mask variable in masked regions. Either 0 or 1.
        fill_value: A float fill value to use inside of masked regions. Can also be
            "mean", in which case the means of the normalizer applied to the
            network's inputs for each batch are used as channel-specific fill
            values.
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
        if isinstance(self.fill_value, int):
            # YAML loads e.g. ``fill_value: 0`` as an int, which would otherwise
            # be mistaken for a fill value mapping.
            self.fill_value = float(self.fill_value)

    def build(self, mask: HasGetSpatialMask) -> "StaticSpatialMasking":
        """
        Build StaticSpatialMasking.

        With ``fill_value="mean"``, the means are passed when the masking is
        called, since they may vary per batch.
        """
        return StaticSpatialMasking(
            mask_value=self.mask_value,
            fill_value=self.fill_value,
            mask=mask,
            exclude=NameAndPrefixMatcher(self.exclude_names_and_prefixes),
        )


class StaticSpatialMasking:
    def __init__(
        self,
        mask_value: int,
        fill_value: float | TensorMapping | Literal["mean"],
        mask: HasGetSpatialMask,
        exclude: NameAndPrefixMatcher = NameAndPrefixMatcher(),
    ):
        """
        Args:
            mask_value: Value of the mask variable in masked regions.
            fill_value: Fill value for masked regions: a float, a mapping from
                variable name to fill value, or "mean" to fill with the means
                passed at call time.
            mask: Provider of the mask for each variable.
            exclude: Names and prefixes of variables which are not masked.
        """
        fill_mapping: TensorMapping | None
        if isinstance(fill_value, float):
            fill_mapping = collections.defaultdict(lambda: torch.as_tensor(fill_value))
        elif isinstance(fill_value, str):  # "mean"
            fill_mapping = None
        else:
            fill_mapping = fill_value
        self._fill_mapping = fill_mapping
        self._mask_value = mask_value
        self._mask = mask
        self._exclude = exclude

    def _masks(self, name: str) -> bool:
        return not self._exclude.match(name)

    def __call__(
        self, data: TensorMapping, means: TensorMapping | None = None
    ) -> TensorDict:
        """
        Apply masking to the data for standard names recognized by a stacker.

        Args:
            data: The data to mask.
            means: Fill values when configured with ``fill_value="mean"``,
                ignored otherwise. Each may be a scalar or a per-sample tensor
                which broadcasts against the data.
        """
        fill_mapping = self._fill_mapping
        if fill_mapping is None:
            if means is None:
                raise ValueError(
                    "StaticSpatialMasking with fill_value 'mean' requires means "
                    "when called."
                )
            fill_mapping = means
        data_: TensorDict = {**data}
        for name, tensor in data_.items():
            if not self._masks(name):
                continue
            mask = self._mask.get_mask_tensor_for(name)
            if mask is None:
                continue
            try:
                fill_value = fill_mapping[name]
            except KeyError as err:
                raise KeyError(
                    "StaticSpatialMasking was initialized with a fill_value mapping "
                    f"but the mapping is missing key '{name}'."
                ) from err
            # broadcast_to, not full_like: means may be per-sample tensors.
            fill = torch.broadcast_to(
                torch.as_tensor(fill_value, dtype=tensor.dtype, device=tensor.device),
                tensor.shape,
            )
            mask = mask.expand(fill.shape)
            masked = replace_on_mask(
                original=tensor,
                replacement=fill,
                mask=mask,
                mask_value=self._mask_value,
            )
            data_[name] = masked
        return data_


class NullSpatialMasking:
    def __call__(
        self, data: TensorMapping, means: TensorMapping | None = None
    ) -> TensorDict:
        return dict(data)


SpatialMasking = StaticSpatialMasking | NullSpatialMasking
"""The type of a spatial masker: it replaces values in masked regions and is
the identity elsewhere (or a no-op when there is no mask). Annotating with
this type, rather than a bare callable, keeps arbitrary data-transforming
functions from flowing into seams that assume masking semantics."""
