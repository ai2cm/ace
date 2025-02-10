import torch

from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector


def test_UNetDiffusionModule_forward_pass():
    downscale_factor = 2
    coarse_shape = (8, 16)
    fine_shape = coarse_shape[0] * downscale_factor, coarse_shape[1] * downscale_factor
    n_channels = 3

    module = DiffusionModuleRegistrySelector(
        "unet_diffusion_song", {"model_channels": 4}
    ).build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        fine_topography=None,
        sigma_data=1.0,
    )

    batch_size = 1
    coarse = torch.randn(batch_size, n_channels, *coarse_shape)
    latent = torch.randn(batch_size, n_channels, *fine_shape)
    noise = torch.randn(batch_size, 1, 1, 1)

    assert (batch_size, n_channels, *fine_shape) == module(latent, coarse, noise).shape
