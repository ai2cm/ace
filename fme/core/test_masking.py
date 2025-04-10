import re
from typing import Optional

import pytest
import torch

import fme
from fme.core.masking import StaticMasking, StaticMaskingConfig

DEVICE = fme.get_device()


class _Mask:
    LEVEL_PATTERN = re.compile(r"_(\d+)$")

    def __init__(
        self,
        mask_2d: Optional[torch.Tensor] = None,
        mask_3d: Optional[torch.Tensor] = None,
    ):
        self.mask_2d = mask_2d
        self.mask_3d = mask_3d

    def get_mask_tensor_for(self, name) -> Optional[torch.Tensor]:
        if name == "mask_ignored":
            return None
        match = self.LEVEL_PATTERN.search(name)
        if match:
            # 3D variable
            if self.mask_3d is None:
                return None
            level = int(match.group(1))
            return self.mask_3d.select(dim=-1, index=level)
        else:
            # 2D variable
            return self.mask_2d


def test_masking_config():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["c"],
        mask_value=1,
        fill_value=0.0,
    )
    _ = config.build(_Mask(mask_2d=torch.ones(1, 1)))
    _ = config.build(_Mask(mask_3d=torch.ones(1, 1, 1)))
    _ = config.build(_Mask(mask_2d=torch.ones(1, 1), mask_3d=torch.ones(1, 1, 1)))

    with pytest.raises(ValueError, match="mask_value must be either 0 or 1"):
        _ = StaticMaskingConfig(
            variable_names_and_prefixes=["c"],
            mask_value=3,
            fill_value=0.0,
        )

    config = StaticMaskingConfig(
        variable_names_and_prefixes=["c"],
        mask_value=1,
        fill_value="mean",
    )
    with pytest.raises(ValueError, match="fill_values mapping required"):
        _ = config.build(_Mask(mask_2d=torch.ones(1, 1)))


_SIZE = (4, 4)

_MASK_2D = torch.ones(_SIZE, device=DEVICE)
_MASK_3D = torch.ones(_SIZE + (2,), device=DEVICE)
_MASK_2D[1, 1] = 0
_MASK_3D[0, :, 0] = 0
_MASK_3D[1, :, 1] = 0


_DATA = {
    "PRESsfc": 10.0 + torch.rand(size=_SIZE, device=DEVICE),
    "specific_total_water_0": torch.rand(size=_SIZE, device=DEVICE),
    "specific_total_water_1": torch.rand(size=_SIZE, device=DEVICE),
}


@pytest.mark.parametrize(
    "var_names_and_prefixes",
    [
        ["PRESsfc", "specific_total_water_"],
        ["PRESsfc", "specific_total_water"],
        ["PRESsfc", "specific_total_water_0"],
        None,
    ],
)
def test_masking(var_names_and_prefixes):
    config = StaticMaskingConfig(
        variable_names_and_prefixes=var_names_and_prefixes,
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(_Mask(_MASK_2D, _MASK_3D))
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    if var_names_and_prefixes and "specific_total_water_0" in var_names_and_prefixes:
        assert torch.all(output["specific_total_water_1"][1, :] != 0.0)
    else:
        assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_with_means():
    config = StaticMaskingConfig(
        mask_value=0,
        fill_value="mean",
    )
    mask = config.build(
        mask=_Mask(_MASK_2D, _MASK_3D),
        means={
            "PRESsfc": torch.tensor(1.0, device=DEVICE),
            "specific_total_water_0": torch.tensor(2.0, device=DEVICE),
            "specific_total_water_1": torch.tensor(3.0, device=DEVICE),
        },
    )
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 1.0
    assert output["PRESsfc"][0, 1] != 1.0
    assert torch.all(output["specific_total_water_0"][0, :] == 2.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 3.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 3.0)


def test_masking_no_3d_masking():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["PRESsfc"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(_Mask(mask_2d=_MASK_2D))
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_no_surface_masking():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["specific_total_water"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(_Mask(mask_3d=_MASK_3D))
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
    mask = config.build(_Mask(mask_3d=_MASK_3D))
    masked = mask(_DATA)
    torch.testing.assert_close(masked["PRESsfc"], _DATA["PRESsfc"])


def test_masking_missing_3d_mask():
    config = StaticMaskingConfig(
        variable_names_and_prefixes=["PRESsfc", "specific_total_water_"],
        mask_value=0,
        fill_value=0.0,
    )
    mask = config.build(_Mask(mask_2d=_MASK_2D))
    masked = mask(_DATA)
    assert masked["PRESsfc"][1, 1] == 0.0
    assert masked["PRESsfc"][0, 1] != 0.0
    for name in ["specific_total_water_0", "specific_total_water_1"]:
        torch.testing.assert_close(masked[name], _DATA[name])


def test_static_masking_error_on_missing_mean():
    mask = StaticMasking(
        mask_value=0,
        fill_value={
            "specific_total_water_0": torch.tensor(2.0, device=DEVICE),
            "specific_total_water_1": torch.tensor(3.0, device=DEVICE),
        },
        mask=_Mask(_MASK_2D, _MASK_3D),
    )
    with pytest.raises(KeyError, match="missing key 'PRESsfc'"):
        _ = mask(_DATA)


def test_static_masking_mask_ignored_name():
    mask = StaticMasking(
        mask_value=0,
        fill_value=float("nan"),
        mask=_Mask(_MASK_2D, _MASK_3D),
    )
    data = {
        "masked": torch.rand(size=_SIZE, device=DEVICE),
        "mask_ignored": torch.rand(size=_SIZE, device=DEVICE),
    }
    masked = mask(data)
    assert masked["masked"][1, 1].isnan()
    # no change to "mask_ignored" because _Mask gives it special treatment
    torch.testing.assert_close(masked["mask_ignored"], data["mask_ignored"])
