import pytest
import torch

from fme.core.device import get_device
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.diffusion.loss import MSELoss, WeightedMappingLoss


@pytest.mark.parametrize("mean", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_WeightedMappingLoss(mean, scale):
    loss = MSELoss()
    n_channels = 5
    packer = Packer([f"var_{i}" for i in range(n_channels)])
    out_names = [f"var_{i}" for i in range(n_channels)]
    normalizer = StandardNormalizer(
        means={name: torch.as_tensor(mean) for name in out_names},
        stds={name: torch.as_tensor(scale) for name in out_names},
    )
    mapping_loss = WeightedMappingLoss(
        loss,
        weights={},
        out_names=out_names,
        normalizer=normalizer,
    )
    x = torch.randn(
        15,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    y = torch.randn(
        15,
        n_channels,
        10,
        10,
    ).to(get_device(), dtype=torch.float)
    x_mapping = {name: x[:, i, :, :] for i, name in enumerate(packer.names)}
    y_mapping = {name: y[:, i, :, :] for i, name in enumerate(packer.names)}
    batch_weights = torch.ones((15, 1, 1, 1), device=get_device())
    assert torch.allclose(
        mapping_loss(x_mapping, y_mapping, batch_weights),
        loss(x, y, batch_weights) / scale**2,
    )
