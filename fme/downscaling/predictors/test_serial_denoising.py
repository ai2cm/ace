from unittest.mock import MagicMock

import pytest
import torch

from fme.downscaling.predictors.serial_denoising import (
    DenoisingMoECheckpointConfig,
    DenoisingRangeModelConfig,
    _SigmaDispatchModule,
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
