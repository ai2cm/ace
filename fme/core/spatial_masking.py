import collections
import dataclasses
from collections.abc import Iterable
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
            "mean", in which case the normalizer means are used as channel-specific
            fill values. Applies to every masked channel that no
            ``fill_value_overrides`` entry selects.
        exclude_names_and_prefixes: Names (2D variables) and prefixes (3D variables)
            to exclude when applying the mask.
        fill_value_overrides: Per-channel fill values overriding ``fill_value``,
            keyed by name (2D variables) or prefix (3D variables), selecting
            channels by the same convention as ``exclude_names_and_prefixes``.
            Values take the same form as ``fill_value``. Where more than one key
            selects a channel the longest key wins, so an exact ``name_<level>``
            beats the ``name_`` prefix, which beats the bare ``name``; a channel
            no key selects takes ``fill_value``. Every key must select at least
            one channel, so a key that selects nothing raises at build time
            rather than silently doing nothing.

            Use this where the right fill differs by channel: the channel mean is
            a neutral filler where masked cells hold no data, whereas ``0.0``
            marks them as out of range, which is usable information where the
            masked cells are genuinely dry.

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

    def _resolve_fill_values(
        self, names: Iterable[str]
    ) -> dict[str, Literal["mean"] | float]:
        """The fill value to use for each of ``names``.

        Override keys select channels by the same name-or-prefix convention as
        ``exclude_names_and_prefixes``. Where more than one key selects a
        channel the longest key wins, so an exact ``name_<level>`` beats the
        ``name_`` prefix, which beats the bare ``name``. A channel no key
        selects takes ``fill_value``.

        Raises:
            ValueError: If an override key selects none of ``names``.
        """
        overrides = self.fill_value_overrides or {}
        matchers = {key: NameAndPrefixMatcher([key]) for key in overrides}
        channels = list(names)
        unselected = sorted(
            key
            for key, matcher in matchers.items()
            if not any(matcher.match(channel) for channel in channels)
        )
        if unselected:
            raise ValueError(
                f"fill_value_overrides keys match no channel: {unselected}. "
                "Keys are names or prefixes, as in exclude_names_and_prefixes."
            )
        resolved: dict[str, Literal["mean"] | float] = {}
        for channel in channels:
            selecting = [
                key for key, matcher in matchers.items() if matcher.match(channel)
            ]
            resolved[channel] = (
                overrides[max(selecting, key=len)] if selecting else self.fill_value
            )
        return resolved

    def build(self, mask: HasGetSpatialMask, means: TensorMapping | None = None):
        """
        Build StaticSpatialMasking.

        """
        exclude = NameAndPrefixMatcher(self.exclude_names_and_prefixes)
        if not self.fill_value_overrides:
            if self.fill_value != "mean":
                fill = torch.as_tensor(float(self.fill_value))
                return StaticSpatialMasking(
                    mask_value=self.mask_value,
                    fill_value=collections.defaultdict(lambda: fill),
                    mask=mask,
                    exclude=exclude,
                )
            if means is None:
                raise ValueError(
                    "fill_values mapping required by build unless configured "
                    "fill_value is a float."
                )
            return StaticSpatialMasking(
                mask_value=self.mask_value,
                fill_value=means,
                mask=mask,
                exclude=exclude,
            )
        # A per-channel mapping cannot be deferred to a defaultdict or to `means`
        # itself, so it is resolved eagerly over the channels `means` names.
        if means is None:
            raise ValueError(
                "fill_values mapping required by build when fill_value_overrides "
                "is set, to enumerate the channels and to supply any 'mean' fills."
            )
        fill_mapping = {
            name: means[name] if value == "mean" else torch.as_tensor(float(value))
            for name, value in self._resolve_fill_values(means).items()
        }
        return StaticSpatialMasking(
            mask_value=self.mask_value,
            fill_value=fill_mapping,
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
