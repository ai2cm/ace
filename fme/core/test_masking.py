import pytest
import torch

import fme
from fme.core.masking import MaskingConfig


def test_masking_config():
    config = MaskingConfig(
        variable_names_and_prefixes=["c"],
        mask_name="a",
        surface_mask_name="b",
        mask_value=1,
        fill_value=0.0,
    )
    mask = config.build()
    assert mask.mask_name == "a"
    assert mask.mask_value == 1
    assert mask.fill_value == 0.0

    with pytest.raises(ValueError) as err:
        _ = MaskingConfig(
            variable_names_and_prefixes=["c"],
            mask_name="a",
            surface_mask_name="b",
            mask_value=3,
            fill_value=0.0,
        )
    assert "mask_value must be either 0 or 1" in str(err.value)


_SIZE = (4, 4)

_MASK_DATA = {
    "surface_mask": torch.ones(_SIZE, device=fme.get_device()),
    "mask_0": torch.ones(_SIZE, device=fme.get_device()),
    "mask_1": torch.ones(_SIZE, device=fme.get_device()),
}
_MASK_DATA["surface_mask"][1, 1] = 0
_MASK_DATA["mask_0"][0, :] = 0
_MASK_DATA["mask_1"][1, :] = 0


_DATA = {
    "PRESsfc": 10.0 + torch.rand(size=_SIZE, device=fme.get_device()),
    "specific_total_water_0": torch.rand(size=_SIZE, device=fme.get_device()),
    "specific_total_water_1": torch.rand(size=_SIZE, device=fme.get_device()),
}


def test_masking():
    config = MaskingConfig(
        variable_names_and_prefixes=["PRESsfc", "specific_total_water_"],
        mask_name="mask",
        surface_mask_name="surface_mask",
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build()
    output = mask(_DATA, _MASK_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_no_3d_masking():
    config = MaskingConfig(
        variable_names_and_prefixes=["PRESsfc"],
        mask_name="surface_mask",
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build()
    output = mask(_DATA, _MASK_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_no_surface_masking():
    config = MaskingConfig(
        variable_names_and_prefixes=["specific_total_water_"],
        mask_name="mask",
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build()
    output = mask(_DATA, _MASK_DATA)
    assert output["PRESsfc"][1, 1] != 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_missing_2d_mask():
    config = MaskingConfig(
        variable_names_and_prefixes=["PRESsfc"],
        mask_name="mask",
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build()
    with pytest.raises(RuntimeError) as err:
        _ = mask(_DATA, _MASK_DATA)
    assert "surface_mask_name is None" in str(err.value)
    assert "PRESsfc" in str(err.value)
