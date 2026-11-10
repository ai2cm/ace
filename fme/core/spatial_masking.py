import collections
import dataclasses
from collections.abc import Iterable
from typing import Literal, Protocol, runtime_checkable

import torch

from fme.core.name_and_prefix_matcher import (
    NameAndPrefixMatcher,
    NameAndPrefixSelection,
)
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
            "mean", in which case the normalizer means are used as channel-specific
            fill values.
        exclude_names_and_prefixes: Names (2D variables) and prefixes (3D variables)
            to exclude when applying the mask.
        fill_value_overrides: Per-channel overrides of ``fill_value``, keyed by
            name or prefix as in ``exclude_names_and_prefixes``. Each key must
            match at least one masked input, and no two keys may match the same
            one.

    """

    mask_value: int
    fill_value: Literal["mean"] | float = 0.0
    exclude_names_and_prefixes: list[str] | None = None
    fill_value_overrides: dict[str, Literal["mean"] | float] | None = None

    def __post_init__(self):
        if self.mask_value not in [0, 1]:
            raise ValueError(
                f"mask_value must be either 0 or 1, but got {self.mask_value}"
            )

    def _resolve_overrides(
        self, names: Iterable[str]
    ) -> dict[str, Literal["mean"] | float]:
        """The override fill value for each of ``names`` that a key matches."""
        overrides = self.fill_value_overrides or {}
        names = list(names)
        unmatched = NameAndPrefixSelection(tuple(overrides)).unmatched_entries(names)
        if unmatched:
            raise ValueError(
                f"fill_value_overrides keys match no masked input: {unmatched}"
            )
        matchers = {key: NameAndPrefixMatcher([key]) for key in overrides}
        resolved: dict[str, Literal["mean"] | float] = {}
        overlapping: set[tuple[str, ...]] = set()
        for name in names:
            keys = [key for key, matcher in matchers.items() if matcher.match(name)]
            if len(keys) > 1:
                overlapping.add(tuple(sorted(keys)))
            elif keys:
                resolved[name] = overrides[keys[0]]
        if overlapping:
            raise ValueError(
                "fill_value_overrides keys match the same input: "
                f"{sorted(overlapping)}"
            )
        return resolved

    def build(
        self,
        mask: HasGetSpatialMask,
        means: TensorMapping | None = None,
        names: Iterable[str] | None = None,
    ):
        """
        Build StaticSpatialMasking.

        Args:
            mask: Provides the mask for each variable.
            means: Normalizer means, required where a "mean" fill applies.
            names: Input names, required when ``fill_value_overrides`` is set.
        """
        exclude = NameAndPrefixMatcher(self.exclude_names_and_prefixes)
        fill_mapping: dict[str, torch.Tensor] = {}
        if self.fill_value_overrides:
            if names is None:
                raise ValueError(
                    "names required by build when fill_value_overrides is set."
                )
            masked = [
                name
                for name in names
                if not exclude.match(name)
                and mask.get_mask_tensor_for(name) is not None
            ]
            for name, value in self._resolve_overrides(masked).items():
                if value != "mean":
                    fill_mapping[name] = torch.as_tensor(float(value))
                elif means is None:
                    raise ValueError(
                        "means required by build for a 'mean' fill_value_overrides "
                        "value."
                    )
                elif name not in means:
                    raise ValueError(
                        f"fill_value_overrides sets a 'mean' fill for {name!r}, "
                        "which has no normalizer mean."
                    )
                else:
                    fill_mapping[name] = means[name]
        if self.fill_value != "mean":
            default = torch.as_tensor(float(self.fill_value))
            return StaticSpatialMasking(
                mask_value=self.mask_value,
                fill_value=collections.defaultdict(lambda: default, fill_mapping),
                mask=mask,
                exclude=exclude,
            )
        if means is None:
            raise ValueError("means required by build when fill_value is 'mean'.")
        return StaticSpatialMasking(
            mask_value=self.mask_value,
            fill_value={**means, **fill_mapping},
            mask=mask,
            exclude=exclude,
        )


class StaticSpatialMasking:
    def __init__(
        self,
        mask_value: int,
        fill_value: float | TensorMapping,
        mask: HasGetSpatialMask,
        exclude: NameAndPrefixMatcher = NameAndPrefixMatcher(),
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
        self._exclude = exclude

    def _masks(self, name: str) -> bool:
        return not self._exclude.match(name)

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
                    "StaticSpatialMasking was initialized with a fill_value mapping "
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


class NullSpatialMasking:
    def __call__(self, data: TensorMapping) -> TensorDict:
        return dict(data)


SpatialMasking = StaticSpatialMasking | NullSpatialMasking
"""The type of a spatial masker: it replaces values in masked regions and is
the identity elsewhere (or a no-op when there is no mask). Annotating with
this type, rather than a bare callable, keeps arbitrary data-transforming
functions from flowing into seams that assume masking semantics."""
