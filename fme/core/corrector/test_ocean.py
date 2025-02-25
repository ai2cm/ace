import datetime

import pytest
import torch

from fme import get_device
from fme.core.gridded_ops import LatLonOperations
from fme.core.masking import StaticMaskingConfig

from .ocean import OceanCorrector, OceanCorrectorConfig


def test_ocean_corrector_init_error():
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    timestep = datetime.timedelta(seconds=3600)

    config = OceanCorrectorConfig()
    # no error
    _ = OceanCorrector(config, ops, None, timestep)

    config = OceanCorrectorConfig(
        masking=StaticMaskingConfig(
            variable_names_and_prefixes=["sst", "so_"],
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

    def get_idepth(self) -> torch.Tensor:
        return torch.tensor([0, 5, 15], device=DEVICE)

    def depth_integral(self, integrand: torch.Tensor) -> torch.Tensor:
        thickness = self.get_idepth().diff(dim=-1)
        return torch.nansum(_MASK * integrand * thickness, dim=-1)


_VERTICAL_COORD = _MockDepth()


def test_ocean_corrector_integration():
    """Ensures that OceanCorrector can be called with all methods active
    but doesn't check results."""
    torch.manual_seed(0)
    config = OceanCorrectorConfig(
        masking=StaticMaskingConfig(
            variable_names_and_prefixes=["sst", "so_"],
            mask_value=0,
            fill_value=float("nan"),
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
        assert corrected_gen[name][_LAT, _LON].isnan().all()
    for name in ["so_0", "so_1"]:
        x = corrected_gen[name].clone()
        x[_LAT, _LON] = 0.0
        assert torch.all(x >= 0.0)
