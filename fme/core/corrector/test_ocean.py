import datetime
import math

import pytest
import torch

from fme import get_device
from fme.core.corrector.ocean import (
    OceanCorrector,
    OceanCorrectorConfig,
    SeaIceFractionConfig,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.masking import StaticMaskingConfig
from fme.core.typing_ import TensorMapping


def test_ocean_corrector_init_error():
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)

    config = OceanCorrectorConfig()
    # no error
    _ = OceanCorrector(config, ops, None, timestep)

    config = OceanCorrectorConfig(
        masking=StaticMaskingConfig(
            mask_value=0,
            fill_value=float("nan"),
        ),
    )
    with pytest.raises(
        ValueError,
        match="OceanCorrector.masking configured but DepthCoordinate missing.",
    ):
        _ = OceanCorrector(config, ops, None, timestep)


DEVICE = get_device()
IMG_SHAPE = (5, 5)
NZ = 2

_MASK = torch.ones(*IMG_SHAPE, NZ, device=DEVICE)
_LAT, _LON = 2, 2
_MASK[_LAT, _LON, :] = 0.0


class _MockDepth:
    def __len__(self) -> int:
        return len(self.get_idepth())

    def get_mask(self) -> torch.Tensor:
        return _MASK

    def get_mask_level(self, level: int):
        return _MASK.select(-1, level)

    def get_mask_tensor_for(self, name: str) -> torch.Tensor | None:
        if name.endswith("_1"):
            return _MASK.select(-1, 1)
        else:
            return _MASK.select(-1, 0)

    def get_idepth(self) -> torch.Tensor:
        return torch.tensor([0, 5, 15], device=DEVICE)

    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        thickness = self.get_idepth().diff(dim=-1)
        return torch.nansum(_MASK * integrand * thickness, dim=-1)

    def to(self, device: str) -> "_MockDepth":
        return self


_VERTICAL_COORD = _MockDepth()


@pytest.mark.parametrize("fill_value", [float("nan"), -1.0, 100.0])
def test_ocean_corrector_integration(fill_value):
    """Ensures that OceanCorrector can be called with all methods active
    but doesn't check results."""
    torch.manual_seed(0)
    config = OceanCorrectorConfig(
        masking=StaticMaskingConfig(
            mask_value=0,
            fill_value=fill_value,
        ),
        force_positive_names=["so_0", "so_1"],
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    corrector = OceanCorrector(config, ops, _VERTICAL_COORD, timestep)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    corrected_gen = corrector(input_data, gen_data, {})
    for name in ["so_0", "so_1", "sst"]:
        if math.isnan(fill_value):
            assert corrected_gen[name][_LAT, _LON].isnan().item()
        else:
            torch.testing.assert_close(
                corrected_gen[name][_LAT, _LON],
                torch.tensor(fill_value, device=DEVICE),
                rtol=0,
                atol=0,
            )
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)


def test_ocean_corrector_has_no_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = OceanCorrector(config, ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(input_data, gen_data, next_step_input_data)
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert not corrected_violation.any()
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_ocean_corrector_has_negative_ocean_fraction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            remove_negative_ocean_fraction=False,
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    input_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    input_data["land_fraction"] = torch.ones(IMG_SHAPE, device=DEVICE) * 0.8
    gen_data = {f"so_{i}": torch.randn(IMG_SHAPE, device=DEVICE) for i in range(NZ)}
    gen_data["sst"] = torch.randn(IMG_SHAPE, device=DEVICE)
    gen_data["sea_ice_fraction"] = torch.randn(IMG_SHAPE, device=DEVICE) * 0.5
    gen_data["sea_ice_fraction"][_LAT, _LON] = -0.5
    corrector = OceanCorrector(config, ops, None, timestep)
    violation = (input_data["land_fraction"] + gen_data["sea_ice_fraction"]) > 1.0
    assert violation.any()
    negative_sea_ice_fraction = gen_data["sea_ice_fraction"] < 0.0
    assert negative_sea_ice_fraction.any()

    next_step_input_data: TensorMapping = {}
    gen_data_corrected = corrector(input_data, gen_data, next_step_input_data)
    corrected_violation = (
        input_data["land_fraction"] + gen_data_corrected["sea_ice_fraction"]
    ) > 1.0
    assert corrected_violation.any()
    # sea_ice_fraction values are still clamped to [0, 1]
    assert not (gen_data_corrected["sea_ice_fraction"] < 0.0).any()


def test_sea_ice_thickness_correction():
    config = OceanCorrectorConfig(
        sea_ice_fraction_correction=SeaIceFractionConfig(
            sea_ice_fraction_name="sea_ice_fraction",
            land_fraction_name="land_fraction",
            sea_ice_thickness_name="HI",
        ),
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)
    input_data = {"land_fraction": torch.ones(IMG_SHAPE, device=DEVICE)}
    input_data["land_fraction"][:3, :3] = torch.rand(3, 3, device=DEVICE)
    gen_data = {
        "sea_ice_fraction": torch.rand(IMG_SHAPE, device=DEVICE),
        "HI": torch.rand(IMG_SHAPE, device=DEVICE) * 10,
    }
    corrector = OceanCorrector(config, ops, None, timestep)
    gen_data_corrected = corrector(input_data, gen_data, {})
    sea_ice_zero = gen_data_corrected["sea_ice_fraction"] == 0.0
    thickness = gen_data_corrected["HI"]
    torch.testing.assert_close(
        torch.where(sea_ice_zero, thickness, 0.0), torch.zeros_like(thickness)
    )
