import abc
import re
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from fme.core.masking import NullMasking, StaticMasking
from fme.core.typing_ import TensorDict, TensorMapping

LEVEL_PATTERN = re.compile(r"_(\d+)$")


class MaskProviderABC(abc.ABC):
    SelfType = TypeVar("SelfType", bound="MaskProviderABC")

    @abc.abstractmethod
    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None: ...

    @abc.abstractmethod
    def to(self: SelfType, device: str) -> SelfType: ...

    @abc.abstractmethod
    def update(self: SelfType, other: SelfType) -> None: ...

    @abc.abstractmethod
    def build_output_masker(self) -> Callable[[TensorMapping], TensorDict]: ...

    @abc.abstractmethod
    def to_state(self) -> dict[str, Any]: ...


class _NullMaskProvider(MaskProviderABC):
    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        return None

    def to(self, device: str) -> "_NullMaskProvider":
        return self

    def update(self, other: MaskProviderABC) -> None:
        if not isinstance(other, _NullMaskProvider):
            raise ValueError(
                f"Attempted to update NullMaskProvider with non-null {other}"
            )

    def build_output_masker(self) -> Callable[[TensorMapping], TensorDict]:
        return NullMasking()

    def to_state(self) -> dict[str, Any]:
        return {"masks": {}}


NullMaskProvider = _NullMaskProvider()


class MaskProvider(MaskProviderABC):
    """
    Stores and returns 2D mask tensors.

    Masks are special time-invariant data variables with names that start with
    the string "mask_". There are three types of masks that MaskProvider knows
    how to use, in order from highest to lowest priority:

    1. Variable-specific masks, e.g. "mask_sst" and "mask_thetao_1".
    2. Level-specific masks, e.g. "mask_0" and "mask_1".
    3. A catch-all 2D mask named "mask_2d".

    For example, when matching the variable "theta_1", MaskProvider will first
    check if it has a mask called "mask_theta_1". If not, then it will check for
    a mask called "mask_1", returning None if this is also not found.

    Another example: when matching the variable "sst", MaskProvider checks for
    "mask_sst" and then "mask_2d", returning None if neither are found.

    Args:
        masks: A dictionary where the keys are required to start with "mask_"
            and values are 2D mask tensors.

    """

    def __init__(self, masks: TensorDict | None = None):
        self._masks = {} if masks is None else masks
        for key in self._masks:
            if not key.startswith("mask_"):
                raise ValueError(
                    "The 'mask' TensorDict passed to MaskProvider init has "
                    f"non-mask tensors, including {key}. Expected all keys "
                    "to start with the string 'mask_'."
                )

    def build_output_masker(self) -> Callable[[TensorMapping], TensorDict]:
        """
        Returns a StaticMasking object that fills in NaNs outside of mask
        valid points, i.e. where the mask value is 0.

        """
        return StaticMasking(
            mask_value=0,
            fill_value=float("nan"),
            mask=self,
        )

    @property
    def masks(self) -> TensorMapping:
        return self._masks

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        # variable specific
        mask_name = f"mask_{name}"
        if mask_name in self.masks:
            return self.masks[mask_name]
        # level specific for 3D vars
        match = LEVEL_PATTERN.search(name)
        if match:
            level = int(match.group(1))
            mask_name = f"mask_{level}"
            return self.masks.get(mask_name, None)
        # 2D mask or None
        return self.masks.get("mask_2d", None)

    def to(self, device: str) -> "MaskProvider":
        return MaskProvider(
            {name: tensor.to(device) for name, tensor in self.masks.items()}
        )

    def update(self, other: "MaskProvider") -> None:
        """Update the MaskProvider's masks with masks from another MaskProvider.

        Raises a ValueError if there are overlapping mask names between the two
        MaskProviders.
        """
        self_keys = set(self.masks.keys())
        other_keys = set(other.masks.keys())
        intersection = self_keys.intersection(other_keys)
        if intersection:
            raise ValueError(
                "Cannot update MaskProvider with overlapping mask names: "
                f"{', '.join(sorted(list(intersection)))}"
            )
        self._masks.update(other.masks)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MaskProvider):
            return False
        if not self.masks.keys() == other.masks.keys():
            return False
        for name, mask in self.masks.items():
            try:
                torch.testing.assert_close(mask, other.masks[name])
            except AssertionError:
                return False
        return True

    def __repr__(self) -> str:
        return (
            "MaskProvider(\n    masks=[\n        "
            + ",\n        ".join(sorted(list(self.masks.keys())))
            + ",\n    ]\n)"
        )

    def to_state(self) -> dict[str, Any]:
        return {"masks": self.masks}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MaskProvider":
        return cls(masks=state["masks"])
