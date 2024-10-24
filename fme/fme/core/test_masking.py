from types import MappingProxyType
from typing import List, Mapping

import pytest
import torch

import fme
from fme.core.climate_data import ClimateData
from fme.core.masking import MaskingConfig
from fme.core.typing_ import TensorMapping

MOCK_FIELD_NAME_PREFIXES = MappingProxyType(
    {
        "specific_total_water": ["specific_total_water_"],
        "surface_pressure": ["PRESsfc"],
        "wetmask": ["wetmask_"],
    }
)


def test_masking_config():
    config = MaskingConfig(
        mask_name="a", mask_value=1, number_of_vertical_levels=5, fill_value=0.0
    )
    mask = config.build()
    assert mask.mask_name == "a"
    assert mask.mask_value == 1
    assert mask.number_of_vertical_levels == 5
    assert mask.fill_value == 0.0


class MockMaskClimateData(ClimateData):
    """This mock climate data class is used for testing masking.
    We expect to make a climate data abstraction class in the future for ocean data"""

    def __init__(
        self,
        mock_data: TensorMapping,
        mock_field_name_prefixes: Mapping[str, List[str]] = MOCK_FIELD_NAME_PREFIXES,
    ):
        super().__init__(
            climate_data=mock_data,
            climate_field_name_prefixes=mock_field_name_prefixes,
        )

    @property
    def wetmask(self) -> torch.Tensor:
        prefix = self._prefixes["wetmask"]
        return self._extract_levels(prefix)


def test_masking():
    config = MaskingConfig(
        mask_name="wetmask",
        mask_value=1,
        number_of_vertical_levels=2,
        fill_value=0.0,
        surface_mask_name="wetmask_0",
    )
    mask = config.build()
    assert sorted(mask.forcing_names) == ["wetmask_0", "wetmask_1"]
    size = (4, 4)
    data = {
        "PRESsfc": 10.0 + torch.rand(size=size, device=fme.get_device()),
        "specific_total_water_0": torch.rand(size=size, device=fme.get_device()),
        "specific_total_water_1": torch.rand(size=size, device=fme.get_device()),
        "wetmask_0": torch.ones(size, device=fme.get_device()),
        "wetmask_1": torch.ones(size, device=fme.get_device()),
    }
    data["wetmask_0"][0, :] = 0
    data["wetmask_1"][1, :] = 0
    output = mask(data, MockMaskClimateData)
    assert torch.all(output["PRESsfc"][0, :] == 0.0)
    assert torch.all(output["PRESsfc"][1, :] != 0.0)
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_with_ocean_fraction_as_surface_mask():
    config = MaskingConfig(
        mask_name="wetmask",
        mask_value=1,
        number_of_vertical_levels=2,
        fill_value=0.0,
        surface_mask_name="ocean_fraction",
    )
    size = (4, 4)
    data = {
        "PRESsfc": 10.0 + torch.rand(size=size, device=fme.get_device()),
        "specific_total_water_0": torch.rand(size=size, device=fme.get_device()),
        "specific_total_water_1": torch.rand(size=size, device=fme.get_device()),
        "wetmask_0": torch.ones(size, device=fme.get_device()),
        "wetmask_1": torch.ones(size, device=fme.get_device()),
    }
    forcing_data = {
        "ocean_fraction": torch.ones(size, device=fme.get_device()),
    }
    forcing_data["ocean_fraction"][1, 1] = 0
    mask = config.build()
    assert sorted(mask.forcing_names) == ["ocean_fraction", "wetmask_0", "wetmask_1"]
    with pytest.raises(ValueError):
        output = mask(data, MockMaskClimateData)
    output = mask(data, MockMaskClimateData, forcing_data)
    assert torch.all(output["PRESsfc"][1, 1] == 0.0)
