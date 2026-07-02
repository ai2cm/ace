import collections
import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import torch

from fme.core.name_and_prefix_matcher import NameAndPrefixMatcher
from fme.core.typing_ import TensorDict, TensorMapping

# ---------------------------------------------------------------------------
# Per-cell loss/eval masks that ride in the batch as ordinary data variables.
#
# Distinct from the static ``SpatialMaskProvider`` masks below (time-invariant,
# resolved at build time for output masking): these are the time-varying (or
# broadcast time-invariant) ``mask_<field>`` validity fields (1 = valid,
# 0 = invalid) loaded alongside the data, applied per timestep in BOTH the
# training loss and the evaluation aggregators. Shared here so the stepper
# (loss) and the aggregators (eval) resolve them identically.
# ---------------------------------------------------------------------------

_MASK_LEVEL_RE = re.compile(r"_?(\d+)$")


def resolve_mask_var_name(
    out_name: str,
    available: set[str],
    explicit_mapping: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the per-cell mask data-variable for output variable ``out_name``.

    Masks ride in the batch as ordinary (time-varying or broadcast
    time-invariant) data variables. Resolution order:

    1. an explicit ``out_name -> mask_var`` mapping (the dataset's
       self-describing per-field ``mask_variable`` attribute, authoritative);
    2. a variable-specific ``mask_<out_name>``;
    3. a level-shared ``mask_<level>`` for a flattened plev field whose name
       ends in the level (e.g. ``ta1000`` -> ``mask_1000``);
    4. a catch-all ``mask_2d``.

    Returns the chosen mask variable name if present in ``available``, else
    ``None`` (the variable is left unmasked).
    """
    if explicit_mapping is not None and out_name in explicit_mapping:
        candidate = explicit_mapping[out_name]
        return candidate if candidate in available else None
    if f"mask_{out_name}" in available:
        return f"mask_{out_name}"
    match = _MASK_LEVEL_RE.search(out_name)
    if match and f"mask_{match.group(1)}" in available:
        return f"mask_{match.group(1)}"
    if "mask_2d" in available:
        return "mask_2d"
    return None


def extract_spatial_masks(
    data: TensorMapping,
    names: Sequence[str],
    explicit_mapping: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor] | None:
    """Per-variable validity masks (1 = valid) for ``names``, pulled from ``data``.

    ``data`` is a batch mapping that carries the ``mask_<field>`` variables as
    ordinary entries (the loader loads them alongside the fields). Each
    requested name is resolved via :func:`resolve_mask_var_name`; the full mask
    tensor is returned unsliced (shape broadcast-compatible with the field,
    typically ``[sample, time, lat, lon]``). Variables without a resolvable
    mask are omitted. Returns ``None`` when no requested variable has a mask, so
    callers can no-op cleanly (identical behavior to the unmasked path).
    """
    available = set(data.keys())
    out: dict[str, torch.Tensor] = {}
    for name in names:
        mask_var = resolve_mask_var_name(name, available, explicit_mapping)
        if mask_var is not None:
            out[name] = data[mask_var]
    return out or None


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
    def build_output_spatial_masker(self) -> Callable[[TensorMapping], TensorDict]: ...

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

    def build(self, mask: HasGetSpatialMask, means: TensorMapping | None = None):
        """
        Build StaticSpatialMasking.

        """
        exclude = NameAndPrefixMatcher(self.exclude_names_and_prefixes)
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
