from unittest.mock import MagicMock

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.predictors.serial_denoising import (
    DenoisingMoECheckpointConfig,
    DenoisingRangeModelConfig,
    _SigmaDispatchModule,
    _validate_experts_compatible,
    _validate_sigma_ranges,
)


def _range(sigma_min: float, sigma_max: float) -> DenoisingRangeModelConfig:
    return DenoisingRangeModelConfig(
        checkpoint_config=MagicMock(),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def test_validate_sigma_ranges_min_ge_max_raises():
    with pytest.raises(ValueError, match="sigma_min < sigma_max"):
        _validate_sigma_ranges([_range(10.0, 10.0)])


def test_validate_sigma_ranges_gap_raises():
    with pytest.raises(ValueError, match="contiguous"):
        _validate_sigma_ranges([_range(0.0, 10.0), _range(11.0, 80.0)])


def test_denoising_moe_config_sorts_ranges():
    cfg = DenoisingMoECheckpointConfig(
        denoising_range_configs=[_range(10.0, 80.0), _range(0.002, 10.0)],
        num_diffusion_generation_steps=10,
    )
    assert cfg.denoising_range_configs[0].sigma_min == 0.002
    assert cfg.denoising_range_configs[1].sigma_min == 10.0


def test_sigma_dispatch_routes_to_correct_expert():
    calls: list[int] = []

    class Expert(torch.nn.Module):
        def __init__(self, which: int):
            super().__init__()
            self.which = which

        def forward(self, x, x_lr, sigma):
            calls.append(self.which)
            return x * 0 + float(self.which)

    e0 = Expert(0)
    e1 = Expert(1)
    dispatch = _SigmaDispatchModule([(0.0, 10.0), (10.0, 20.0)], [e0, e1])

    x = torch.ones(1, 1, 4, 4)
    lr = torch.ones(1, 1, 4, 4)
    out_high = dispatch(x, lr, torch.tensor(5.0))
    assert out_high.mean().item() == 0.0
    out_low = dispatch(x, lr, torch.tensor(15.0))
    assert out_low.mean().item() == 1.0
    assert calls == [0, 1]


def test_sigma_dispatch_boundary_prefers_lower_sigma_range():
    calls: list[int] = []

    class Expert(torch.nn.Module):
        def __init__(self, which: int):
            super().__init__()
            self.which = which

        def forward(self, x, x_lr, sigma):
            calls.append(self.which)
            return x * 0 + float(self.which)

    e0 = Expert(0)
    e1 = Expert(1)
    dispatch = _SigmaDispatchModule([(0.0, 10.0), (10.0, 20.0)], [e0, e1])
    x = torch.ones(1, 1, 4, 4)
    out = dispatch(x, x, torch.tensor(10.0))
    assert out.mean().item() == 0.0
    assert calls == [0]


def test_sigma_dispatch_out_of_range_raises():
    class Expert(torch.nn.Module):
        def forward(self, x, x_lr, sigma):
            return x

    dispatch = _SigmaDispatchModule([(0.0, 10.0)], [Expert()])
    with pytest.raises(ValueError, match="not covered"):
        dispatch(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2), torch.tensor(11.0))


def _mock_expert(static_inputs=None):
    expert = MagicMock()
    expert.in_packer.names = ["a", "b"]
    expert.out_packer.names = ["c"]
    expert.coarse_shape = (4, 8)
    expert.downscale_factor = 4
    expert.config.predict_residual = False
    expert.static_inputs = static_inputs
    return expert


def test_validate_experts_compatible_mixed_static_inputs_raises():
    e0 = _mock_expert(static_inputs=None)
    e1 = _mock_expert(static_inputs=MagicMock())
    with pytest.raises(ValueError, match="static_inputs"):
        _validate_experts_compatible([e0, e1])


def test_validate_experts_compatible_mismatched_static_coords_raises():
    si0 = MagicMock()
    si0.coords = LatLonCoordinates(
        lat=torch.linspace(-90, 90, 16), lon=torch.linspace(0, 360, 32)
    )
    si1 = MagicMock()
    si1.coords = LatLonCoordinates(
        lat=torch.linspace(-90, 90, 8), lon=torch.linspace(0, 360, 16)
    )
    e0 = _mock_expert(static_inputs=si0)
    e1 = _mock_expert(static_inputs=si1)
    with pytest.raises(ValueError, match="static_inputs coordinates"):
        _validate_experts_compatible([e0, e1])


def test_validate_experts_compatible_matching_static_inputs():
    coords = LatLonCoordinates(
        lat=torch.linspace(-90, 90, 16), lon=torch.linspace(0, 360, 32)
    )
    si0 = MagicMock()
    si0.coords = coords
    si1 = MagicMock()
    si1.coords = coords
    e0 = _mock_expert(static_inputs=si0)
    e1 = _mock_expert(static_inputs=si1)
    _validate_experts_compatible([e0, e1])


def test_validate_experts_compatible_both_none_static_inputs():
    e0 = _mock_expert(static_inputs=None)
    e1 = _mock_expert(static_inputs=None)
    _validate_experts_compatible([e0, e1])
