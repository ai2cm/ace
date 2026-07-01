import re

import pytest
import torch

import fme
from fme.core.spatial_masking import (
    StaticSpatialMasking,
    StaticSpatialMaskingConfig,
    extract_spatial_masks,
    resolve_mask_var_name,
)

DEVICE = fme.get_device()


def test_resolve_mask_var_name_precedence():
    available = {"mask_ta1000", "mask_1000", "mask_2d"}
    # explicit mapping wins
    assert (
        resolve_mask_var_name("ta1000", available, {"ta1000": "mask_1000"})
        == "mask_1000"
    )
    # explicit mapping to an absent mask -> None
    assert resolve_mask_var_name("ta1000", available, {"ta1000": "nope"}) is None
    # mask_<var> beats mask_<level>
    assert resolve_mask_var_name("ta1000", available) == "mask_ta1000"
    # mask_<level> from trailing digits when no mask_<var>
    assert resolve_mask_var_name("ua1000", available) == "mask_1000"
    # catch-all mask_2d
    assert resolve_mask_var_name("pr", available) == "mask_2d"
    # nothing resolves
    assert resolve_mask_var_name("pr", {"mask_1000"}) is None


def test_extract_spatial_masks():
    m1000 = torch.ones(2, 3, 4, 8)
    data = {
        "ta1000": torch.randn(2, 3, 4, 8),
        "ua1000": torch.randn(2, 3, 4, 8),
        "pr": torch.randn(2, 3, 4, 8),
        "mask_1000": m1000,
    }
    masks = extract_spatial_masks(data, ["ta1000", "ua1000", "pr"])
    # ta1000 + ua1000 share mask_1000 (level convention); pr has no mask
    assert set(masks) == {"ta1000", "ua1000"}
    assert masks["ta1000"] is m1000 and masks["ua1000"] is m1000
    # explicit mapping honored
    masks2 = extract_spatial_masks(data, ["ta1000"], {"ta1000": "mask_1000"})
    assert masks2["ta1000"] is m1000
    # no masks present -> None (clean no-op for callers)
    assert extract_spatial_masks({"pr": torch.randn(1)}, ["pr"]) is None


class _Mask:
    LEVEL_PATTERN = re.compile(r"_(\d+)$")

    def __init__(
        self,
        mask_2d: torch.Tensor | None = None,
        mask_3d: torch.Tensor | None = None,
    ):
        self.mask_2d = mask_2d
        self.mask_3d = mask_3d

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
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

    def build_output_spatial_masker(self):
        raise NotImplementedError

    def to(self, device: str) -> "_Mask":
        return self


def test_masking_config():
    config = StaticSpatialMaskingConfig(
        mask_value=1,
        fill_value=0.0,
    )
    _ = config.build(_Mask(mask_2d=torch.ones(1, 1)))
    _ = config.build(_Mask(mask_3d=torch.ones(1, 1, 1)))
    _ = config.build(_Mask(mask_2d=torch.ones(1, 1), mask_3d=torch.ones(1, 1, 1)))

    with pytest.raises(ValueError, match="mask_value must be either 0 or 1"):
        _ = StaticSpatialMaskingConfig(
            mask_value=3,
            fill_value=0.0,
        )

    config = StaticSpatialMaskingConfig(
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
    "exclude",
    [
        None,
        [],
        ["specific_total_water_1"],
    ],
)
def test_masking(exclude):
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=0.0,
        exclude_names_and_prefixes=exclude,
    )
    mask = config.build(_Mask(_MASK_2D, _MASK_3D))
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    if exclude and "specific_total_water_1" in exclude:
        assert torch.all(output["specific_total_water_1"][1, :] != 0.0)
    else:
        assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


@pytest.mark.parametrize(
    "exclude",
    [
        "PRESsfc",
        "specific_total_water_",
        "specific_total_water",
        "specific_total_water_0",
    ],
)
def test_masking_exclusion(exclude):
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=float("nan"),
        exclude_names_and_prefixes=[exclude],
    )
    mask = config.build(mask=_Mask(_MASK_2D, _MASK_3D))
    output = mask(_DATA)
    if exclude == "PRESsfc":
        assert torch.all(~output["PRESsfc"].isnan())
        assert torch.all(output["specific_total_water_0"][0, :].isnan())
        assert torch.all(output["specific_total_water_1"][1, :].isnan())
    else:
        assert output["PRESsfc"][1, 1].isnan()
    if exclude == "specific_total_water_0":
        assert torch.all(~output["specific_total_water_0"].isnan())
        assert torch.all(output["specific_total_water_1"][1, :].isnan())
    elif exclude != "PRESsfc":
        assert torch.all(~output["specific_total_water_0"].isnan())
        assert torch.all(~output["specific_total_water_1"].isnan())


def test_masking_with_means():
    config = StaticSpatialMaskingConfig(
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
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=0.0,
        exclude_names_and_prefixes=["specific_total_water"],
    )
    mask = config.build(_Mask(mask_2d=_MASK_2D))
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] == 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] != 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_no_surface_masking():
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=0.0,
        exclude_names_and_prefixes=["PRESsfc"],
    )
    mask = config.build(_Mask(mask_3d=_MASK_3D))
    output = mask(_DATA)
    assert output["PRESsfc"][1, 1] != 0.0
    assert output["PRESsfc"][0, 1] != 0.0
    assert torch.all(output["specific_total_water_0"][0, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][1, :] == 0.0)
    assert torch.all(output["specific_total_water_1"][0, :] != 0.0)


def test_masking_missing_2d_mask():
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=0.0,
        exclude_names_and_prefixes=["specific_total_water"],
    )
    mask = config.build(_Mask(mask_3d=_MASK_3D))
    masked = mask(_DATA)
    torch.testing.assert_close(masked["PRESsfc"], _DATA["PRESsfc"])


def test_masking_missing_3d_mask():
    config = StaticSpatialMaskingConfig(
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
    mask = StaticSpatialMasking(
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
    mask = StaticSpatialMasking(
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
