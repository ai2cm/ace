import pytest
import torch

from fme.core.device import get_device
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector


@pytest.mark.parametrize("use_topography", [True, False])
def test_diffusion_unet_shapes(use_topography):
    downscale_factor = 2
    coarse_shape = (9, 18)
    fine_shape = (18, 36)
    fine_topography = torch.ones(*fine_shape)

    n_channels = 3

    unet = (
        DiffusionModuleRegistrySelector(
            "unet_diffusion_song",
            dict(
                model_channels=4,
            ),
        )
        .build(
            n_in_channels=n_channels,
            n_out_channels=n_channels,
            coarse_shape=coarse_shape,
            downscale_factor=downscale_factor,
            fine_topography=fine_topography if use_topography else None,
            sigma_data=1.0,
        )
        .to(get_device())
    )

    batch_size = 2
    coarse = torch.randn(batch_size, n_channels, *coarse_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise_level = torch.randn(batch_size, 1, 1, 1)
    outputs = unet(latent, coarse, noise_level)

    assert (batch_size, n_channels, *fine_shape) == outputs.shape
