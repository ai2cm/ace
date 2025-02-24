import pytest
import torch

import fme
from fme.core.masking import StaticMasking, StaticMaskingConfig


def test_masking_config():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["c"],
        mask_value=1,
        fill_value=0.0,
    )
    _ = config.build(mask_2d=torch.ones(1, 1))
    _ = config.build(mask_3d=torch.ones(1, 1, 1))
    _ = config.build(mask_2d=torch.ones(1, 1), mask_3d=torch.ones(1, 1, 1))

    with pytest.raises(ValueError, match="mask_value must be either 0 or 1"):
        _ = StaticMaskingConfig(
            variable_names_and_prefixes=["c"],
            mask_value=3,
            fill_value=0.0,
        )


def test_masking_init_errors():
    with pytest.raises(ValueError, match="mask_2d or mask_3d must be provided"):
        _ = StaticMasking(
            variable_names_and_prefixes=["c"],
            mask_value=3,
            fill_value=0.0,
        )
    with pytest.raises(ValueError, match="mask_2d with 2 dimensions"):
        _ = StaticMasking(
            variable_names_and_prefixes=["c"],
            mask_value=3,
            fill_value=0.0,
            mask_2d=torch.ones(1, 1, 1),
        )
    with pytest.raises(ValueError, match="mask_3d with 3 dimensions"):
        _ = StaticMasking(
            variable_names_and_prefixes=["c"],
            mask_value=3,
            fill_value=0.0,
            mask_3d=torch.ones(1, 1),
        )


_SIZE = (4, 4)

_MASK_2D = torch.ones(_SIZE, device=fme.get_device())
_MASK_3D = torch.ones(_SIZE + (2,), device=fme.get_device())
_MASK_2D[1, 1] = 0
_MASK_3D[0, :, 0] = 0
_MASK_3D[1, :, 1] = 0


_DATA = {
    "PRESsfc": 10.0 + torch.rand(size=_SIZE, device=fme.get_device()),
    "specific_total_water_0": torch.rand(size=_SIZE, device=fme.get_device()),
    "specific_total_water_1": torch.rand(size=_SIZE, device=fme.get_device()),
}


def test_masking():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["PRESsfc", "specific_total_water_"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(_MASK_2D, _MASK_3D)
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_no_3d_masking():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["PRESsfc"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(mask_2d=_MASK_2D)
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_no_surface_masking():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["specific_total_water_"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(mask_3d=_MASK_3D)
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] != 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_missing_2d_mask():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["PRESsfc"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(mask_3d=_MASK_3D)
    with pytest.raises(RuntimeError, match="PRESsfc is a 2D variable"):
        _ = mask(_DATA)


def test_masking_missing_3d_mask():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["PRESsfc", "specific_total_water_"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(mask_2d=_MASK_2D)
    with pytest.raises(RuntimeError, match="specific_total_water is a 3D variable"):
        _ = mask(_DATA)
