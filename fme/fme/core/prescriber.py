import dataclasses
from typing import Dict, List
import torch


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for overwriting the predictions of 'prescribed_name' by the target
    values in the region where 'mask_name' == 'mask_value'."""

    prescribed_name: str
    mask_name: str
    mask_value: int

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
        )


class Prescriber:
    """
    Responsible for overwriting model predictions by target data in masked regions.
    """

    def __init__(self, prescribed_name: str, mask_name: str, mask_value: int):
        self.prescribed_name = prescribed_name
        self.mask_name = mask_name
        self.mask_value = mask_value

    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        gen_norm: Dict[str, torch.Tensor],
        target_norm: Dict[str, torch.Tensor],
    ):
        # overwrite specified generated variable in given mask region
        prescribed_gen = torch.where(
            data[self.mask_name] == self.mask_value,
            target_norm[self.prescribed_name],
            gen_norm[self.prescribed_name],
        )
        return {**gen_norm, self.prescribed_name: prescribed_gen}

    def get_state(self):
        return {
            "prescribed_name": self.prescribed_name,
            "mask_name": self.mask_name,
            "mask_value": self.mask_value,
        }

    def load_state(self, state):
        self.prescribed_name = state["prescribed_name"]
        self.mask_name = state["mask_name"]
        self.mask_value = state["mask_value"]

    @classmethod
    def from_state(cls, state) -> "Prescriber":
        return Prescriber(
            state["prescribed_name"], state["mask_name"], state["mask_value"]
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
