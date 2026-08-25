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
            "mean", in which case the normalizer means are used as channel-specific
            fill values. Applies to every masked channel that no
            ``fill_value_overrides`` entry selects.
        exclude_names_and_prefixes: Names (2D variables) and prefixes (3D variables)
            to exclude when applying the mask.
        fill_value_overrides: Per-channel fill values overriding ``fill_value``,
            keyed by name (2D variables) or prefix (3D variables), following the
            same convention as ``exclude_names_and_prefixes``. Values take the same
            form as ``fill_value``. A channel selected by an exact name uses that
            entry; otherwise the longest matching prefix wins; otherwise
            ``fill_value`` applies. Every key must select at least one channel, so
            a typo raises at build time rather than silently doing nothing.

            The motivating case is that the right fill differs by channel. A field
            whose masked cells hold no data (temperature, salinity) has no true
            value to restore, so the channel mean is a neutral filler. A field
            whose masked cells are genuinely dry -- the depth-resolved channels
            below the sea floor -- carries usable information in being marked out
            of range, which the mean erases.

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
        for key, value in (self.fill_value_overrides or {}).items():
            if value != "mean" and not isinstance(value, int | float):
                raise ValueError(
                    "fill_value_overrides values must be a float or 'mean', but "
                    f"got {value!r} for {key!r}"
                )

    def _resolve_fill_value(self, name: str) -> Literal["mean"] | float:
        """The fill value for one channel.

        Exact name wins, then the longest matching prefix, then ``fill_value``.
        """
        overrides = self.fill_value_overrides or {}
        if name in overrides:
            return overrides[name]
        prefixes = [k for k in overrides if name.startswith(k)]
        if prefixes:
            return overrides[max(prefixes, key=len)]
        return self.fill_value

    def build(self, mask: HasGetSpatialMask, means: TensorMapping | None = None):
        """
        Build StaticSpatialMasking.

        """
        exclude = NameAndPrefixMatcher(self.exclude_names_and_prefixes)
        if not self.fill_value_overrides:
            # Unchanged from before per-channel overrides existed.
            if isinstance(self.fill_value, float):
                return StaticSpatialMasking(
                    mask_value=self.mask_value,
                    fill_value=collections.defaultdict(
                        lambda: torch.as_tensor(self.fill_value)
                    ),
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
        # With overrides the mapping is resolved per channel, so it has to be
        # built eagerly over the known channel names rather than deferred to a
        # defaultdict or to `means` itself.
        if means is None:
            raise ValueError(
                "fill_values mapping required by build when fill_value_overrides "
                "is set, to enumerate the channels and to supply any 'mean' fills."
            )
        unmatched = [
            key
            for key in self.fill_value_overrides
            if not any(name == key or name.startswith(key) for name in means)
        ]
        if unmatched:
            raise ValueError(
                "fill_value_overrides keys match no channel: "
                f"{sorted(unmatched)}. Keys are names or prefixes, as in "
                "exclude_names_and_prefixes."
            )
        fill_mapping = {}
        for name in means:
            resolved = self._resolve_fill_value(name)
            fill_mapping[name] = (
                means[name] if resolved == "mean" else torch.as_tensor(float(resolved))
            )
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
