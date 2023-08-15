import dataclasses
from typing import Dict, List

import torch


@dataclasses.dataclass
class PrescriberConfig:
    """
    Configuration for overwriting predictions of 'prescribed_name' by target values.

    If interpolate is False, the data is overwritten in the region where
    'mask_name' == 'mask_value' after values are cast to integer. If interpolate
    is True, the data is interpolated between the predicted value at 0 and the
    target value at 1 based on the mask variable, and it is assumed the mask variable
    lies in the range from 0 to 1.

    Attributes:
        prescribed_name: Name of the variable to be overwritten.
        mask_name: Name of the mask variable.
        mask_value: Value of the mask variable in the region to be overwritten.
        interpolate: Whether to interpolate linearly between the generated and target
            values in the masked region, where 0 means keep the generated values and
            1 means replace completely with the target values. Requires mask_value
            be set to 1.
    """

    prescribed_name: str
    mask_name: str
    mask_value: int
    interpolate: bool = False

    def __post_init__(self):
        if self.interpolate and self.mask_value != 1:
            raise ValueError(
                "Interpolation requires mask_value to be 1, but it is set to "
                f"{self.mask_value}."
            )

    def build(self, in_names: List[str], out_names: List[str]):
        if not (self.prescribed_name in in_names and self.prescribed_name in out_names):
            raise ValueError(
                "Variables which are being prescribed in masked regions must be in"
                f" in_names and out_names, but {self.prescribed_name} is not."
            )
        return Prescriber(
            prescribed_name=self.prescribed_name,
            mask_name=self.mask_name,
            mask_value=self.mask_value,
            interpolate=self.interpolate,
        )


class Prescriber:
    """
    Responsible for overwriting model predictions by target data in masked regions.
    """

    def __init__(
        self,
        prescribed_name: str,
        mask_name: str,
        mask_value: int,
        interpolate: bool = False,
    ):
        self.prescribed_name = prescribed_name
        self.mask_name = mask_name
        self.mask_value = mask_value
        self.interpolate = interpolate

    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        gen_norm: Dict[str, torch.Tensor],
        target_norm: Dict[str, torch.Tensor],
    ):
        """
        Args:
            data: Dictionary of data containing the mask variable.
            gen_norm: Dictionary of generated data.
            target_norm: Dictionary of target data.
        """
        if self.interpolate:
            mask = data[self.mask_name]
            # 0 keeps the generated values, 1 replaces completely with the target values
            prescribed_gen = (
                mask * target_norm[self.prescribed_name]
                + (1 - mask) * gen_norm[self.prescribed_name]
            )
        else:
            # overwrite specified generated variable in given mask region
            prescribed_gen = torch.where(
                torch.round(data[self.mask_name]).to(int) == self.mask_value,
                target_norm[self.prescribed_name],
                gen_norm[self.prescribed_name],
            )
        return {**gen_norm, self.prescribed_name: prescribed_gen}

    def get_state(self):
        return {
            "prescribed_name": self.prescribed_name,
            "mask_name": self.mask_name,
            "mask_value": self.mask_value,
            "interpolate": self.interpolate,
        }

    def load_state(self, state):
        self.prescribed_name = state["prescribed_name"]
        self.mask_name = state["mask_name"]
        self.mask_value = state["mask_value"]
        interpolate = state.get("interpolate", False)
        self.interpolate = interpolate

    @classmethod
    def from_state(cls, state) -> "Prescriber":
        return Prescriber(
            state["prescribed_name"],
            state["mask_name"],
            state["mask_value"],
            state.get("interpolate", False),
        )


class NullPrescriber:
    """Dummy prescriber that does nothing."""

    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        gen_norm: Dict[str, torch.Tensor],
        target_norm: Dict[str, torch.Tensor],
    ):
        return gen_norm

    def get_state(self):
        return {}

    def load_state(self, state):
        return
