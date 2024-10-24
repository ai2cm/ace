import dataclasses
from typing import Callable, List, Optional

import torch

import fme

from .climate_data import ClimateData
from .prescriber import Prescriber
from .typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class MaskingConfig:
    """
    Configuration for applying masking to the generated output.

    Attributes:
        mask_name: The name of the mask in the dataset.
        mask_value: Value of the mask variable in masked regions. Either 0 or 1.
        number_of_vertical_levels: The number of vertical levels in the dataset.
        fill_value: The constant fill value to use outside of masked regions.
        external_mask_path: If specified, read the mask from the NetCDF file,
            otherwise it is assumed that the mask can be found in the input data.
        surface_mask_name: The name of the surface mask in the dataset
            or `ocean_fraction`. If ocean_fraction,
            surface mask will use ocean fraction from forcing data.
    """

    mask_name: str = "mask"
    mask_value: int = 1
    number_of_vertical_levels: int = 1
    fill_value: float = 0.0
    surface_mask_name: str = "ocean_fraction"

    def __post_init__(self):
        if self.mask_value not in [0, 1]:
            raise ValueError(
                "mask_value must be either 0 or 1, but got " f"{self.mask_value}"
            )

    def build(self):
        return Masking(
            mask_name=self.mask_name,
            mask_value=self.mask_value,
            number_of_vertical_levels=self.number_of_vertical_levels,
            fill_value=self.fill_value,
            surface_mask_name=self.surface_mask_name,
        )


class Masking:
    def __init__(
        self,
        mask_name: str,
        mask_value: int,
        number_of_vertical_levels: int,
        fill_value: float,
        surface_mask_name: str,
    ):
        self.mask_name = mask_name
        self.mask_value = mask_value
        self.number_of_vertical_levels = number_of_vertical_levels
        self.fill_value = fill_value
        self.surface_mask_name = surface_mask_name

    @property
    def forcing_names(self) -> List[str]:
        names = [self.surface_mask_name]
        for i in range(self.number_of_vertical_levels):
            names.append(f"{self.mask_name}_{i}")

        return list(set(names))

    def _split(self, output: torch.Tensor, keys: List[str]) -> TensorDict:
        if len(keys) == 1:
            return {keys[0]: output}
        # split the output tensor along the vertical dimension
        tensors = torch.split(output, 1, dim=-1)
        if len(keys) != len(tensors):
            raise ValueError(
                f"Expected {len(keys)} keys, but got {len(tensors)} tensors."
            )
        return {key: tensor.squeeze(dim=-1) for key, tensor in zip(keys, tensors)}

    def __call__(
        self,
        data: TensorMapping,
        climate_data_cls: Callable[[TensorMapping], ClimateData],
        forcing_data: Optional[TensorMapping] = None,
    ) -> TensorDict:
        """
        Attributes:
            data: Generated data
            climate_data_cls: ClimateData type class
            forcing_data: This is required if ocean fraction is the surface mask
        Returns:
            masked_gen: Masked generated data
        """
        gen_data: ClimateData = climate_data_cls(data)
        mask = getattr(gen_data, self.mask_name)
        mask_3d = {self.mask_name: mask.to(fme.get_device())}
        if self.surface_mask_name == "ocean_fraction":
            if forcing_data is None:
                raise ValueError(
                    "Forcing data must be provided when using \
                        ocean_fraction as the surface mask."
                )
            ocean_fraction = forcing_data[self.surface_mask_name].to(fme.get_device())
            mask_2d = {self.mask_name: ocean_fraction}
        else:
            mask_2d = {
                self.mask_name: gen_data.data[self.surface_mask_name].to(
                    fme.get_device()
                )
            }

        masked_gen: TensorDict = {}
        for name in gen_data.standard_names:
            try:
                gen_3d = {name: gen_data[name]}
            except KeyError:
                continue
            fill_3d = {name: torch.full_like(gen_3d[name], self.fill_value)}
            vertical_level_keys = gen_data.get_all_level_names(name)
            is_2d = len(vertical_level_keys) == 1
            mask_data = mask_2d if is_2d else mask_3d
            prescriber = Prescriber(name, self.mask_name, 1 - self.mask_value)
            output = prescriber(mask_data, gen_3d, fill_3d)[name]
            masked_gen = {**masked_gen, **self._split(output, vertical_level_keys)}
        return masked_gen
