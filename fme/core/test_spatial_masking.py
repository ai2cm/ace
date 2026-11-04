import re

import dacite
import pytest
import torch

import fme
from fme.core.spatial_masking import StaticSpatialMasking, StaticSpatialMaskingConfig

DEVICE = fme.get_device()


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
    with pytest.raises(ValueError, match="means required"):
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


def _overrides_setup():
    """A 2x2 grid with one masked cell; ``sos`` must not match key ``so``."""
    mask_2d = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=DEVICE)
    mask_3d = mask_2d[..., None].expand(2, 2, 2)
    names = ["sst", "sos", "so_0", "thetao_0", "thetao_1", "deptho"]
    data = {name: torch.full((1, 2, 2), 7.0, device=DEVICE) for name in names}
    means = {
        name: torch.tensor(100.0 * (i + 1), device=DEVICE)
        for i, name in enumerate(names)
    }
    return _Mask(mask_2d=mask_2d, mask_3d=mask_3d), data, means


# the one masked cell of every channel in _overrides_setup
_MASKED_CELL = (0, 0, 1)
_UNMASKED_CELL = (0, 0, 0)


def test_fill_value_overrides_apply_per_channel():
    """mean everywhere except the prefixed and named overrides, which take 0.0."""
    mask, data, means = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value="mean",
        fill_value_overrides={"thetao_": 0.0, "deptho": 0.0},
    )
    out = config.build(mask=mask, means=means, names=data)(data)
    expected = {
        "sst": means["sst"].item(),  # default "mean"
        "sos": means["sos"].item(),  # default "mean"
        "so_0": means["so_0"].item(),  # default "mean"
        "thetao_0": 0.0,  # prefix override
        "thetao_1": 0.0,  # prefix override
        "deptho": 0.0,  # exact-name override
    }
    for name, value in expected.items():
        assert out[name][_MASKED_CELL].item() == pytest.approx(value), name
    # unmasked cells are untouched in every channel
    for name in data:
        assert out[name][_UNMASKED_CELL].item() == pytest.approx(7.0), name


@pytest.mark.parametrize("key", ["so", "so_"])
def test_fill_value_overrides_key_does_not_select_similar_name(key):
    """``so``/``so_`` select ``so_<level>``, not ``sos``."""
    mask, data, means = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value="mean",
        fill_value_overrides={key: 0.0},
    )
    out = config.build(mask=mask, means=means, names=data)(data)
    assert out["so_0"][_MASKED_CELL].item() == pytest.approx(0.0)
    assert out["sos"][_MASKED_CELL].item() == pytest.approx(means["sos"].item())


@pytest.mark.parametrize(
    "overrides",
    [
        {"thetao_": 1.0, "thetao_1": "mean"},  # prefix and exact name
        {"thetao": 1.0, "thetao_": 2.0},  # bare name and prefix
    ],
)
def test_fill_value_overrides_overlapping_keys_raise(overrides):
    mask, data, means = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0, fill_value=0.0, fill_value_overrides=overrides
    )
    with pytest.raises(ValueError, match="match the same input"):
        config.build(mask=mask, means=means, names=data)


@pytest.mark.parametrize(
    "key",
    [
        "thetoa_",  # transposed letters
        "dept",  # a truncation that is a raw string prefix of "deptho"
    ],
)
def test_fill_value_overrides_unmatched_key_raises(key):
    mask, data, means = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value="mean",
        fill_value_overrides={key: 0.0},
    )
    with pytest.raises(ValueError, match="match no masked input"):
        config.build(mask=mask, means=means, names=data)


@pytest.mark.parametrize(
    "key, exclude",
    [
        ("sst", ["sst"]),  # excluded input
        ("mask_ignored", None),  # input with no mask
        ("output_only", None),  # in means but not an input
    ],
)
def test_fill_value_overrides_key_selecting_no_masked_input_raises(key, exclude):
    mask, data, means = _overrides_setup()
    extra = {name: torch.tensor(1.0) for name in ["mask_ignored", "output_only"]}
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value="mean",
        exclude_names_and_prefixes=exclude,
        fill_value_overrides={key: 0.0},
    )
    with pytest.raises(ValueError, match="match no masked input"):
        config.build(mask=mask, means={**means, **extra}, names=[*data, "mask_ignored"])


def test_fill_value_overrides_float_fill_value_needs_no_means():
    mask, data, _ = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=3.5,
        fill_value_overrides={"thetao_": 0.0},
    )
    out = config.build(mask=mask, means=None, names=data)(data)
    assert out["sst"][_MASKED_CELL].item() == pytest.approx(3.5)
    assert out["thetao_0"][_MASKED_CELL].item() == pytest.approx(0.0)


def test_fill_value_overrides_mean_override_requires_means():
    mask, data, _ = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=0.0,
        fill_value_overrides={"thetao_": "mean"},
    )
    with pytest.raises(ValueError, match="means required"):
        config.build(mask=mask, means=None, names=data)


def test_fill_value_overrides_requires_names():
    mask, _, means = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0,
        fill_value=0.0,
        fill_value_overrides={"thetao_": 0.0},
    )
    with pytest.raises(ValueError, match="names required"):
        config.build(mask=mask, means=means)


def test_empty_fill_value_overrides_acts_as_unset():
    mask, data, _ = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0, fill_value=0.0, fill_value_overrides={}
    )
    out = config.build(mask=mask, means=None)(data)
    for name in data:
        assert out[name][_MASKED_CELL].item() == pytest.approx(0.0), name
        assert out[name][_UNMASKED_CELL].item() == pytest.approx(7.0), name


def test_fill_value_overrides_bad_value_rejected_on_load():
    """dacite rejects a non-float, non-"mean" value, so build never sees one."""
    with pytest.raises(dacite.UnionMatchError):
        dacite.from_dict(
            data_class=StaticSpatialMaskingConfig,
            data={
                "mask_value": 0,
                "fill_value": "mean",
                "fill_value_overrides": {"thetao_": "zero"},
            },
            config=dacite.Config(strict=True),
        )


@pytest.mark.parametrize("fill_value", [0.0, 0, 3.5, "mean"])
@pytest.mark.parametrize("overrides", [None, {"deptho": 1.0}])
def test_fill_value_applies_to_channels_no_override_selects(fill_value, overrides):
    """Both build branches agree, including for an integer (YAML ``0``)."""
    mask, data, means = _overrides_setup()
    config = StaticSpatialMaskingConfig(
        mask_value=0, fill_value=fill_value, fill_value_overrides=overrides
    )
    out = config.build(mask=mask, means=means, names=data)(data)
    for name in data:
        if overrides is not None and name in overrides:
            continue
        expected = means[name].item() if fill_value == "mean" else float(fill_value)
        assert out[name][_MASKED_CELL].item() == pytest.approx(expected), name
        assert out[name][_UNMASKED_CELL].item() == pytest.approx(7.0), name
